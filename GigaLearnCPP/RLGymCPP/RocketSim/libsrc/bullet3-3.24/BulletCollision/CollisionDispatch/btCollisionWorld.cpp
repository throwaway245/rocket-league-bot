/*
Bullet Continuous Collision Detection and Physics Library
Copyright (c) 2003-2006 Erwin Coumans  https://bulletphysics.org

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it freely,
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/

#include "btCollisionWorld.h"
#include "btCollisionDispatcher.h"
#include "../CollisionDispatch/btCollisionObject.h"
#include "../CollisionShapes/btCollisionShape.h"
#include "../CollisionShapes/btConvexShape.h"
#include "../NarrowPhaseCollision/btGjkEpaPenetrationDepthSolver.h"
#include "../CollisionShapes/btSphereShape.h"                 //for raycasting
#include "../CollisionShapes/btBvhTriangleMeshShape.h"        //for raycasting
#include "../NarrowPhaseCollision/btRaycastCallback.h"
#include "../CollisionShapes/btCompoundShape.h"
#include "../NarrowPhaseCollision/btSubSimplexConvexCast.h"
#include "../NarrowPhaseCollision/btGjkConvexCast.h"
#include "../NarrowPhaseCollision/btContinuousConvexCollision.h"
#include "../BroadphaseCollision/btCollisionAlgorithm.h"
#include "../BroadphaseCollision/btBroadphaseInterface.h"
#include "../BroadphaseCollision/btDbvt.h"
#include "../../LinearMath/btAabbUtil2.h"
#include "../../LinearMath/btQuickprof.h"
#include "../CollisionShapes/btConvexPolyhedron.h"
#include "../CollisionDispatch/btCollisionObjectWrapper.h"

//#define DISABLE_DBVT_COMPOUNDSHAPE_RAYCAST_ACCELERATION

//#define USE_BRUTEFORCE_RAYBROADPHASE 1
//RECALCULATE_AABB is slower, but benefit is that you don't need to call 'stepSimulation'  or 'updateAabbs' before using a rayTest
//#define RECALCULATE_AABB_RAYCAST 1

//When the user doesn't provide dispatcher or broadphase, create basic versions (and delete them in destructor)
#include "../CollisionDispatch/btCollisionDispatcher.h"
#include "../CollisionDispatch/btCollisionConfiguration.h"

///for debug drawing

//for debug rendering
#include "../CollisionShapes/btBoxShape.h"
#include "../CollisionShapes/btCompoundShape.h"
#include "../CollisionShapes/btPolyhedralConvexShape.h"
#include "../CollisionShapes/btSphereShape.h"
#include "../CollisionShapes/btTriangleCallback.h"
#include "../CollisionShapes/btTriangleMeshShape.h"
#include "../CollisionShapes/btStaticPlaneShape.h"
#include <atomic>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <string.h>

void btCollisionWorld::setup(btCollisionDispatcher* dispatcher, btBroadphaseInterface* pairCache, btCollisionConfiguration* collisionConfiguration)
{
	m_dispatcher1 = dispatcher;
	m_broadphasePairCache = pairCache;
	m_forceUpdateAllAabbs = true;
}

btCollisionWorld::~btCollisionWorld()
{
	//clean up remaining objects
	int i;
	for (i = 0; i < m_collisionObjects.size(); i++)
	{
		btCollisionObject* collisionObject = m_collisionObjects[i];

		btBroadphaseProxy* bp = collisionObject->getBroadphaseHandle();
		if (bp)
		{
			//
			// only clear the cached algorithms
			//
			getBroadphase()->getOverlappingPairCache()->cleanProxyFromPairs(bp, m_dispatcher1);
			getBroadphase()->destroyProxy(bp, m_dispatcher1);
			collisionObject->setBroadphaseHandle(0);
		}
	}
}

void btCollisionWorld::refreshBroadphaseProxy(btCollisionObject* collisionObject)
{
	if (collisionObject->getBroadphaseHandle())
	{
		int collisionFilterGroup = collisionObject->getBroadphaseHandle()->m_collisionFilterGroup;
		int collisionFilterMask = collisionObject->getBroadphaseHandle()->m_collisionFilterMask;

		getBroadphase()->destroyProxy(collisionObject->getBroadphaseHandle(), getDispatcher());

		//calculate new AABB
		btTransform trans = collisionObject->getWorldTransform();

		btVector3 minAabb;
		btVector3 maxAabb;
		collisionObject->getCollisionShape()->getAabb(trans, minAabb, maxAabb);

		int type = collisionObject->getCollisionShape()->getShapeType();
		collisionObject->setBroadphaseHandle(getBroadphase()->createProxy(
			minAabb,
			maxAabb,
			type,
			collisionObject,
			collisionFilterGroup,
			collisionFilterMask,
			m_dispatcher1));
	}
}

void btCollisionWorld::addCollisionObject(btCollisionObject* collisionObject, int collisionFilterGroup, int collisionFilterMask)
{
	btAssert(collisionObject);

	//check that the object isn't already added
	btAssert(m_collisionObjects.findLinearSearch(collisionObject) == m_collisionObjects.size());
	btAssert(collisionObject->getWorldArrayIndex() == -1);  // do not add the same object to more than one collision world

	collisionObject->setWorldArrayIndex(m_collisionObjects.size());
	m_collisionObjects.push_back(collisionObject);

	//calculate new AABB
	btTransform trans = collisionObject->getWorldTransform();

	btVector3 minAabb;
	btVector3 maxAabb;
	collisionObject->getCollisionShape()->getAabb(trans, minAabb, maxAabb);

	int type = collisionObject->getCollisionShape()->getShapeType();
	collisionObject->setBroadphaseHandle(getBroadphase()->createProxy(
		minAabb,
		maxAabb,
		type,
		collisionObject,
		collisionFilterGroup,
		collisionFilterMask,
		m_dispatcher1));
}

void btCollisionWorld::updateSingleAabb(btCollisionObject* colObj)
{
	btVector3 minAabb, maxAabb;
	colObj->getCollisionShape()->getAabb(colObj->getWorldTransform(), minAabb, maxAabb);
	//need to increase the aabb for contact thresholds
	btVector3 contactThreshold(gContactBreakingThreshold, gContactBreakingThreshold, gContactBreakingThreshold);
	minAabb -= contactThreshold;
	maxAabb += contactThreshold;

	if (getDispatchInfo().m_useContinuous && colObj->getInternalType() == btCollisionObject::CO_RIGID_BODY && !colObj->isStaticOrKinematicObject())
	{
		btVector3 minAabb2, maxAabb2;
		colObj->getCollisionShape()->getAabb(colObj->getInterpolationWorldTransform(), minAabb2, maxAabb2);
		minAabb2 -= contactThreshold;
		maxAabb2 += contactThreshold;
		minAabb.setMin(minAabb2);
		maxAabb.setMax(maxAabb2);
	}

	btBroadphaseInterface* bp = (btBroadphaseInterface*)m_broadphasePairCache;

	//moving objects should be moderately sized, probably something wrong if not
	if (colObj->isStaticObject() || ((maxAabb - minAabb).length2() < btScalar(1e12)))
	{
		bp->setAabb(colObj->getBroadphaseHandle(), minAabb, maxAabb, m_dispatcher1);
	}
	else
	{
		//something went wrong, investigate
		//this assert is unwanted in 3D modelers (danger of loosing work)
		colObj->setActivationState(DISABLE_SIMULATION);
		btAssert(false);
	}
}

void btCollisionWorld::updateAabbs()
{
	BT_PROFILE("updateAabbs");

	for (int i = 0; i < m_collisionObjects.size(); i++)
	{
		btCollisionObject* colObj = m_collisionObjects[i];
		btAssert(colObj->getWorldArrayIndex() == i);

		//only update aabb of active objects
		if (m_forceUpdateAllAabbs || colObj->isActive())
		{
			updateSingleAabb(colObj);
		}
	}
}

void btCollisionWorld::computeOverlappingPairs()
{
	BT_PROFILE("calculateOverlappingPairs");
	m_broadphasePairCache->calculateOverlappingPairs(m_dispatcher1);
}

void btCollisionWorld::performDiscreteCollisionDetection()
{
	BT_PROFILE("performDiscreteCollisionDetection");

	btDispatcherInfo& dispatchInfo = getDispatchInfo();

	updateAabbs();

	computeOverlappingPairs();

	btCollisionDispatcher* dispatcher = getDispatcher();
	{
		BT_PROFILE("dispatchAllCollisionPairs");
		if (dispatcher)
			dispatcher->dispatchAllCollisionPairs(m_broadphasePairCache->getOverlappingPairCache(), dispatchInfo, m_dispatcher1);
	}
}

void btCollisionWorld::removeCollisionObject(btCollisionObject* collisionObject)
{
	//bool removeFromBroadphase = false;

	{
		btBroadphaseProxy* bp = collisionObject->getBroadphaseHandle();
		if (bp)
		{
			//
			// only clear the cached algorithms
			//
			getBroadphase()->getOverlappingPairCache()->cleanProxyFromPairs(bp, m_dispatcher1);
			getBroadphase()->destroyProxy(bp, m_dispatcher1);
			collisionObject->setBroadphaseHandle(0);
		}
	}

	int iObj = collisionObject->getWorldArrayIndex();
	//    btAssert(iObj >= 0 && iObj < m_collisionObjects.size()); // trying to remove an object that was never added or already removed previously?
	if (iObj >= 0 && iObj < m_collisionObjects.size())
	{
		btAssert(collisionObject == m_collisionObjects[iObj]);
		m_collisionObjects.swap(iObj, m_collisionObjects.size() - 1);
		m_collisionObjects.pop_back();
		if (iObj < m_collisionObjects.size())
		{
			m_collisionObjects[iObj]->setWorldArrayIndex(iObj);
		}
	}
	else
	{
		// slow linear search
		//swapremove
		m_collisionObjects.remove(collisionObject);
	}
	collisionObject->setWorldArrayIndex(-1);
}

void btCollisionWorld::rayTestSingle(const btTransform& rayFromTrans, const btTransform& rayToTrans,
									 btCollisionObject* collisionObject,
									 const btCollisionShape* collisionShape,
									 const btTransform& colObjWorldTransform,
									 RayResultCallback& resultCallback)
{
	btCollisionObjectWrapper colObWrap(0, collisionShape, collisionObject, colObjWorldTransform, -1, -1);
	btCollisionWorld::rayTestSingleInternal(rayFromTrans, rayToTrans, &colObWrap, resultCallback);
}

// The shape types of the objects that the ray tests consider, and the time that their
// ray tests took. They are only collected when GGL_RAY_STATS is set.
struct RayShapeStats
{
	std::atomic<uint64_t> counts[32];
	std::atomic<uint64_t> ns[32];

	RayShapeStats()
	{
		for (int i = 0; i < 32; i++)
		{
			counts[i].store(0);
			ns[i].store(0);
		}
	}

	~RayShapeStats()
	{
		uint64_t total = 0;
		for (int i = 0; i < 32; i++)
			total += counts[i].load();
		if (total == 0)
			return;
		printf("[btCollisionWorld] ray test candidates: %llu\n", (unsigned long long)total);
		for (int i = 0; i < 32; i++)
		{
			if (counts[i].load() == 0)
				continue;
			printf("[btCollisionWorld]   shape type %2d: %10llu candidates, %8.3f s\n",
				i, (unsigned long long)counts[i].load(), ns[i].load() * 1e-9);
		}
		fflush(stdout);
	}
};

static RayShapeStats g_RayShapeStats;

// Times the ray test of one candidate, so that the time can be told apart per shape type
struct RayShapeStatsTimer
{
	uint64_t start = 0;
	int shapeType = -1;

	RayShapeStatsTimer() {}

	~RayShapeStatsTimer()
	{
		if (shapeType < 0)
			return;
		uint64_t elapsed = (uint64_t)std::chrono::steady_clock::now().time_since_epoch().count() - start;
		g_RayShapeStats.ns[shapeType].fetch_add(elapsed, std::memory_order_relaxed);
	}
};


static bool btRayStatsEnabled()
{
	static const bool enabled = []()
	{
		const char* value = getenv("GGL_RAY_STATS");
		return value != NULL && value[0] != '\0' && value[0] != '0';
	}();
	return enabled;
}

// The direct ray test of a static plane is used unless GGL_DIRECT_PLANE=0 turns it off.
static bool btRayDirectPlaneEnabled()
{
	static const bool enabled = []()
	{
		const char* value = getenv("GGL_DIRECT_PLANE");
		return !(value != NULL && value[0] == '0');
	}();
	return enabled;
}

// The comparison of the direct ray test of a static plane with the generic solver of
// Bullet is only done when GGL_VERIFY_PLANE is set.
static bool btRayVerifyPlaneEnabled()
{
	static const bool enabled = []()
	{
		const char* value = getenv("GGL_VERIFY_PLANE");
		return value != NULL && value[0] != '\0' && value[0] != '0';
	}();
	return enabled;
}

// The hit of the direct ray test of a static plane.
struct btStaticPlaneHit
{
	btScalar m_fraction;
	btVector3 m_normalWorld;
	int m_triangleIndex;
	// The distance of the hit from the diagonal of the square of the triangles and the
	// radius of that square.
	btScalar m_diagonalDistance;
	btScalar m_radius;
};

// The transpose of the basis times the vector, the way that the inverse of a transform
// applies it.
static btVector3 btBasisTransposeTimes(const btMatrix3x3& basis, const btVector3& v)
{
	return btVector3(basis.tdotx(v), basis.tdoty(v), basis.tdotz(v));
}

// The exact test of the generic path for one of its two triangles: the ray has to cross
// the plane of the triangle, and the crossing point has to be inside the triangle. It is
// used for the rays that hit close to the diagonal of the square, where the generic path
// lets both triangles accept the hit and keeps the one with the smaller distance.
static bool btTriangleRayHit(const btVector3& vertex0, const btVector3& vertex1, const btVector3& vertex2,
							  const btVector3& rayFromLocal, const btVector3& rayToLocal,
							  unsigned int rayFlags, btScalar closestHitFraction,
							  btScalar& distanceOut)
{
	const btVector3 v10 = vertex1 - vertex0;
	const btVector3 v20 = vertex2 - vertex0;
	const btVector3 triangleNormal = v10.cross(v20);

	const btScalar dist = vertex0.dot(triangleNormal);
	btScalar distFrom = triangleNormal.dot(rayFromLocal) - dist;
	btScalar distTo = triangleNormal.dot(rayToLocal) - dist;

	if (distFrom * distTo >= btScalar(0.0))
		return false;

	if ((rayFlags & btTriangleRaycastCallback::kF_FilterBackfaces) && (distFrom <= btScalar(0.0)))
		return false;

	const btScalar projLength = distFrom - distTo;
	const btScalar distance = distFrom / projLength;
	if (!(distance < closestHitFraction))
		return false;

	// The edges of the triangle accept a hit with a tolerance of 0.01% of the size of the
	// triangle, the same way as in btTriangleRaycastCallback::processTriangle.
	btScalar edgeTolerance = triangleNormal.length2();
	edgeTolerance *= btScalar(-0.0001);

	btVector3 point;
	point.setInterpolate3(rayFromLocal, rayToLocal, distance);
	const btVector3 v0p = vertex0 - point;
	const btVector3 v1p = vertex1 - point;
	const btVector3 cp0 = v0p.cross(v1p);
	if (!((btScalar)(cp0.dot(triangleNormal)) >= edgeTolerance))
		return false;
	const btVector3 v2p = vertex2 - point;
	const btVector3 cp1 = v1p.cross(v2p);
	if (!((btScalar)(cp1.dot(triangleNormal)) >= edgeTolerance))
		return false;
	const btVector3 cp2 = v2p.cross(v0p);
	if (!((btScalar)(cp2.dot(triangleNormal)) >= edgeTolerance))
		return false;

	distanceOut = distance;
	return true;
}

// The direct ray test of a btStaticPlaneShape.
//
// The generic path of a concave shape builds two triangles around the ray and intersects
// the ray with them (btStaticPlaneShape::processAllTriangles). The equation of the plane
// gives the same hit with much less work: the fraction of the crossing follows from the
// plane equation, and the two triangles always cover the crossing point, so they only
// decide which of the two of them is the hit.
static bool btRayTestStaticPlane(const btStaticPlaneShape* planeShape,
								  const btTransform& colObjWorldTransform,
								  const btVector3& rayFromWorld,
								  const btVector3& rayToWorld,
								  unsigned int rayFlags,
								  btScalar closestHitFraction,
								  btStaticPlaneHit& hitOut)
{
	const btVector3& planeNormal = planeShape->getPlaneNormal();
	const btScalar planeConstant = planeShape->getPlaneConstant();
	const btMatrix3x3& basis = colObjWorldTransform.getBasis();

	// The ray in the local frame of the shape, the same way as the generic path does it.
	const btTransform worldToCollisionObject = colObjWorldTransform.inverse();
	const btVector3 rayFromLocal = worldToCollisionObject * rayFromWorld;
	const btVector3 rayToLocal = worldToCollisionObject * rayToWorld;

	// The square that the two triangles of the generic path cover. Its size and its place
	// follow from the bounding box of the ray in the local frame, the same way as in
	// btStaticPlaneShape::processAllTriangles.
	btVector3 rayAabbMinLocal = rayFromLocal;
	rayAabbMinLocal.setMin(rayToLocal);
	btVector3 rayAabbMaxLocal = rayFromLocal;
	rayAabbMaxLocal.setMax(rayToLocal);

	const btVector3 halfExtents = (rayAabbMaxLocal - rayAabbMinLocal) * btScalar(0.5);
	const btScalar radius = halfExtents.length();
	const btVector3 center = (rayAabbMaxLocal + rayAabbMinLocal) * btScalar(0.5);

	btVector3 tangentDir0, tangentDir1;
	btPlaneSpace1(planeNormal, tangentDir0, tangentDir1);

	const btVector3 projectedCenter = center - (planeNormal.dot(center) - planeConstant) * planeNormal;

	// The first of the two triangles of the generic path.
	const btVector3 vertex0 = projectedCenter + tangentDir0 * radius + tangentDir1 * radius;
	const btVector3 vertex1 = projectedCenter + tangentDir0 * radius - tangentDir1 * radius;
	const btVector3 vertex2 = projectedCenter - tangentDir0 * radius - tangentDir1 * radius;

	// The plane of the triangle and the distances of the two ends of the ray to it, in
	// the same arithmetic as btTriangleRaycastCallback::processTriangle. The distances are
	// the ones of the generic path, so that a ray that ends on the plane is decided the
	// same way by both of them.
	const btVector3 triangleNormal = (vertex1 - vertex0).cross(vertex2 - vertex0);
	const btScalar triangleDist = vertex0.dot(triangleNormal);
	const btScalar distFrom = triangleNormal.dot(rayFromLocal) - triangleDist;
	const btScalar distTo = triangleNormal.dot(rayToLocal) - triangleDist;

	// A ray that keeps the sign of its distances does not cross the plane.
	if (distFrom * distTo >= btScalar(0.0))
		return false;

	// The ray hits the back of the triangle if it comes from the side that the normal of
	// the shape points to.
	if ((rayFlags & btTriangleRaycastCallback::kF_FilterBackfaces) && (distFrom <= btScalar(0.0)))
		return false;

	btScalar fraction = distFrom / (distFrom - distTo);
	if (fraction >= closestHitFraction)
		return false;

	// The two triangles cover the square that is centred on the projected center of the
	// ray and that is as wide as the ray is long. The crossing point is always inside
	// that square, and the diagonal of the square decides which of the two triangles is
	// the hit. The edges of the triangles are accepted with a tolerance, which is
	// 0.02% of the radius of the square.
	btVector3 point;
	point.setInterpolate3(rayFromLocal, rayToLocal, fraction);
	const btVector3 offset = point - projectedCenter;
	const btScalar a = tangentDir0.dot(offset);
	const btScalar b = tangentDir1.dot(offset);

	const btScalar tolerance = radius * btScalar(1.0002);
	const btScalar diagonalTolerance = radius * btScalar(0.0002);
	const bool inFirstTriangle = (a <= tolerance) && (b >= -tolerance) && (a - b >= -diagonalTolerance);
	const bool inSecondTriangle = (a >= -tolerance) && (b <= tolerance) && (b - a >= -diagonalTolerance);
	if (!inFirstTriangle && !inSecondTriangle)
		return false;

	int triangleIndex = inFirstTriangle ? 0 : 1;
	if (inFirstTriangle && inSecondTriangle)
	{
		// Close to the diagonal of the square both triangles accept the hit, and the
		// generic path keeps the one with the smaller distance. That is rare, and the
		// generic path does not answer it stably either, so it is decided the same way
		// as the generic path decides it.
		const btVector3 secondVertex0 = projectedCenter - tangentDir0 * radius - tangentDir1 * radius;
		const btVector3 secondVertex1 = projectedCenter - tangentDir0 * radius + tangentDir1 * radius;
		const btVector3 secondVertex2 = projectedCenter + tangentDir0 * radius + tangentDir1 * radius;

		btScalar firstDistance = btScalar(0.0);
		btScalar secondDistance = btScalar(0.0);
		const bool firstHit = btTriangleRayHit(vertex0, vertex1, vertex2, rayFromLocal, rayToLocal,
											   rayFlags, closestHitFraction, firstDistance);
		const bool secondHit = btTriangleRayHit(secondVertex0, secondVertex1, secondVertex2, rayFromLocal, rayToLocal,
												rayFlags, closestHitFraction, secondDistance);
		if (!firstHit && !secondHit)
			return false;
		// The first triangle wins a tie, because the generic path reports it first and
		// only accepts the second one when its distance is the smaller of the two.
		if (firstHit && (!secondHit || !(secondDistance < firstDistance)))
		{
			triangleIndex = 0;
			fraction = firstDistance;
		}
		else
		{
			triangleIndex = 1;
			fraction = secondDistance;
		}
	}

	hitOut.m_fraction = fraction;
	hitOut.m_triangleIndex = triangleIndex;
	hitOut.m_diagonalDistance = a - b;
	hitOut.m_radius = radius;

	// The normal of the hit points back to the side that the ray came from, the same way
	// as the generic path: the normal of the triangle, flipped when the ray comes from
	// the side that the normal of the shape points to. Both triangles have the same
	// normal, so the sign does not depend on which of them is the hit.
	btVector3 normalLocal = triangleNormal;
	normalLocal.normalize();
	if ((rayFlags & btTriangleRaycastCallback::kF_KeepUnflippedNormal) == 0 && distFrom <= btScalar(0.0))
		normalLocal = -normalLocal;
	hitOut.m_normalWorld = basis * normalLocal;

	return true;
}

// The bits of a positive double keep their order, so the largest of them is found by
// comparing the bits.
static uint64_t btDoubleToBits(double value)
{
	uint64_t bits = 0;
	memcpy(&bits, &value, sizeof(bits));
	return bits;
}

static double btDoubleFromBits(uint64_t bits)
{
	double value = 0;
	memcpy(&value, &bits, sizeof(value));
	return value;
}

static void btAtomicMax(std::atomic<uint64_t>& target, uint64_t value)
{
	uint64_t current = target.load(std::memory_order_relaxed);
	while (value > current && !target.compare_exchange_weak(current, value, std::memory_order_relaxed, std::memory_order_relaxed))
	{
	}
}

// The verification of the direct ray test of a static plane against the generic solver
// of Bullet.
struct PlaneVerificationStats
{
	std::atomic<uint64_t> rays;
	std::atomic<uint64_t> hits;
	std::atomic<uint64_t> hitMismatches;
	std::atomic<uint64_t> fractionMismatches;
	std::atomic<uint64_t> pointMismatches;
	std::atomic<uint64_t> normalMismatches;
	std::atomic<uint64_t> metadataMismatches;
	std::atomic<uint64_t> unstableMetadata;
	std::atomic<uint64_t> maxFractionDifference;
	std::atomic<uint64_t> maxPointDifference;
	std::atomic<uint64_t> maxNormalDifference;
	std::atomic<uint64_t> maxMetadataMismatchOffset;

	PlaneVerificationStats()
	{
		rays.store(0);
		hits.store(0);
		hitMismatches.store(0);
		fractionMismatches.store(0);
		pointMismatches.store(0);
		normalMismatches.store(0);
		metadataMismatches.store(0);
		unstableMetadata.store(0);
		maxFractionDifference.store(0);
		maxPointDifference.store(0);
		maxNormalDifference.store(0);
		maxMetadataMismatchOffset.store(0);
	}

	~PlaneVerificationStats()
	{
		const uint64_t rayCount = rays.load();
		if (rayCount == 0)
			return;
		const uint64_t hitCount = hits.load();
		const uint64_t hitMismatchCount = hitMismatches.load();
		const uint64_t fractionMismatchCount = fractionMismatches.load();
		const uint64_t pointMismatchCount = pointMismatches.load();
		const uint64_t normalMismatchCount = normalMismatches.load();
		const uint64_t metadataMismatchCount = metadataMismatches.load();
		const uint64_t mismatchCount = hitMismatchCount + fractionMismatchCount + pointMismatchCount +
									   normalMismatchCount + metadataMismatchCount;
		printf("[btCollisionWorld] plane verification: %llu rays, %llu hits, %llu mismatches "
			   "(hit %llu, fraction %llu, point %llu, normal %llu, metadata %llu, unstable %llu), "
			   "max fraction difference %g, max point difference %g, max normal difference %g, "
			   "max metadata mismatch offset %g\n",
			   (unsigned long long)rayCount, (unsigned long long)hitCount, (unsigned long long)mismatchCount,
			   (unsigned long long)hitMismatchCount, (unsigned long long)fractionMismatchCount,
			   (unsigned long long)pointMismatchCount, (unsigned long long)normalMismatchCount,
			   (unsigned long long)metadataMismatchCount,
			   (unsigned long long)unstableMetadata.load(),
			   btDoubleFromBits(maxFractionDifference.load()), btDoubleFromBits(maxPointDifference.load()),
			   btDoubleFromBits(maxNormalDifference.load()), btDoubleFromBits(maxMetadataMismatchOffset.load()));
		fflush(stdout);
	}
};

static PlaneVerificationStats g_PlaneVerificationStats;

// The generic path of Bullet, repeated here so that a ray on which the two paths
// disagree can be reported with the numbers that both of them computed. It is only
// used when the verification finds such a ray.
static void btDumpPlaneMismatch(const btStaticPlaneShape* planeShape,
								const btTransform& colObjWorldTransform,
								const btVector3& rayFromWorld,
								const btVector3& rayToWorld,
								bool directHitFound)
{
	static int dumpCount = 0;
	if (dumpCount >= 8)
		return;
	dumpCount++;

	const btVector3& planeNormal = planeShape->getPlaneNormal();
	const btScalar planeConstant = planeShape->getPlaneConstant();
	const btMatrix3x3& basis = colObjWorldTransform.getBasis();
	const btVector3& origin = colObjWorldTransform.getOrigin();

	// The numbers of the direct ray test.
	const btVector3 worldNormal = basis * planeNormal;
	const btScalar worldConstant = planeConstant + worldNormal.dot(origin);
	const btScalar distFrom = worldNormal.dot(rayFromWorld) - worldConstant;
	const btScalar distTo = worldNormal.dot(rayToWorld) - worldConstant;

	// The square of the triangles of the generic path.
	const btTransform worldToCollisionObject = colObjWorldTransform.inverse();
	const btVector3 rayFromLocal = worldToCollisionObject * rayFromWorld;
	const btVector3 rayToLocal = worldToCollisionObject * rayToWorld;
	btVector3 aabbMin = rayFromLocal;
	aabbMin.setMin(rayToLocal);
	btVector3 aabbMax = rayFromLocal;
	aabbMax.setMax(rayToLocal);
	const btVector3 halfExtents = (aabbMax - aabbMin) * btScalar(0.5);
	const btScalar radius = halfExtents.length();
	const btVector3 center = (aabbMax + aabbMin) * btScalar(0.5);
	btVector3 tangentDir0, tangentDir1;
	btPlaneSpace1(planeNormal, tangentDir0, tangentDir1);
	const btVector3 projectedCenter = center - (planeNormal.dot(center) - planeConstant) * planeNormal;

	printf("[btCollisionWorld] plane mismatch: ray (%g %g %g) -> (%g %g %g), plane normal (%g %g %g) "
		   "constant %g, direct hit %d\n",
		   rayFromWorld.x(), rayFromWorld.y(), rayFromWorld.z(),
		   rayToWorld.x(), rayToWorld.y(), rayToWorld.z(),
		   planeNormal.x(), planeNormal.y(), planeNormal.z(), planeConstant, (int)directHitFound);
	printf("[btCollisionWorld]   direct: distFrom %g distTo %g  world constant %g\n",
		   distFrom, distTo, worldConstant);
	printf("[btCollisionWorld]   generic: radius %g center (%g %g %g) projected (%g %g %g)\n",
		   radius, center.x(), center.y(), center.z(),
		   projectedCenter.x(), projectedCenter.y(), projectedCenter.z());

	for (int t = 0; t < 2; t++)
	{
		btVector3 triangle[3];
		if (t == 0)
		{
			triangle[0] = projectedCenter + tangentDir0 * radius + tangentDir1 * radius;
			triangle[1] = projectedCenter + tangentDir0 * radius - tangentDir1 * radius;
			triangle[2] = projectedCenter - tangentDir0 * radius - tangentDir1 * radius;
		}
		else
		{
			triangle[0] = projectedCenter - tangentDir0 * radius - tangentDir1 * radius;
			triangle[1] = projectedCenter - tangentDir0 * radius + tangentDir1 * radius;
			triangle[2] = projectedCenter + tangentDir0 * radius + tangentDir1 * radius;
		}

		const btVector3 v10 = triangle[1] - triangle[0];
		const btVector3 v20 = triangle[2] - triangle[0];
		const btVector3 triangleNormal = v10.cross(v20);
		const btScalar dist = triangle[0].dot(triangleNormal);
		const btScalar distA = triangleNormal.dot(rayFromLocal) - dist;
		const btScalar distB = triangleNormal.dot(rayToLocal) - dist;
		const btScalar projLength = distA - distB;
		const btScalar distance = distA / projLength;

		printf("[btCollisionWorld]   triangle %d: distA %g distB %g distance %g\n", t, distA, distB, distance);
		if (!(distA * distB < btScalar(0.0)) || !(distance < btScalar(1.0)))
			continue;

		btScalar edgeTolerance = triangleNormal.length2();
		edgeTolerance *= btScalar(-0.0001);
		btVector3 point;
		point.setInterpolate3(rayFromLocal, rayToLocal, distance);
		const btVector3 v0p = triangle[0] - point;
		const btVector3 v1p = triangle[1] - point;
		const btVector3 cp0 = v0p.cross(v1p);
		const btVector3 v2p = triangle[2] - point;
		const btVector3 cp1 = v1p.cross(v2p);
		const btVector3 cp2 = v2p.cross(v0p);
		printf("[btCollisionWorld]     edge tests %g %g %g, tolerance %g\n",
			   cp0.dot(triangleNormal), cp1.dot(triangleNormal), cp2.dot(triangleNormal), edgeTolerance);
	}
	fflush(stdout);
}

// Compares the direct ray test of a static plane with the generic one of Bullet, which
// intersects the two triangles that the shape builds around the ray.
static void btVerifyStaticPlaneRay(const btStaticPlaneShape* planeShape,
								   const btTransform& colObjWorldTransform,
								   const btVector3& rayFromWorld,
								   const btVector3& rayToWorld,
								   unsigned int rayFlags,
								   btScalar closestHitFraction,
								   bool directHitFound,
								   const btStaticPlaneHit& directHit)
{
	struct RecordingTriangleCallback : public btTriangleRaycastCallback
	{
		bool m_hit;
		btScalar m_fraction;
		btVector3 m_normalLocal;
		int m_partId;
		int m_triangleIndex;

		RecordingTriangleCallback(const btVector3& from, const btVector3& to, unsigned int flags)
			: btTriangleRaycastCallback(from, to, flags),
			  m_hit(false),
			  m_fraction(btScalar(1.)),
			  m_partId(-1),
			  m_triangleIndex(-1)
		{
		}

		virtual btScalar reportHit(const btVector3& hitNormalLocal, btScalar hitFraction, int partId, int triangleIndex)
		{
			// The result callback of the generic path keeps the hit with the smallest
			// fraction, and the first one wins a tie. The recording does the same, so
			// that the hit that it compares is the hit that the generic path keeps.
			if (!m_hit || hitFraction < m_fraction)
			{
				m_hit = true;
				m_fraction = hitFraction;
				m_normalLocal = hitNormalLocal;
				m_partId = partId;
				m_triangleIndex = triangleIndex;
			}
			return hitFraction;
		}
	};

	// The generic path of a concave shape, the same way that
	// btCollisionWorld::rayTestSingleInternal does it.
	const btTransform worldToCollisionObject = colObjWorldTransform.inverse();
	const btVector3 rayFromLocal = worldToCollisionObject * rayFromWorld;
	const btVector3 rayToLocal = worldToCollisionObject * rayToWorld;

	btVector3 rayAabbMinLocal = rayFromLocal;
	rayAabbMinLocal.setMin(rayToLocal);
	btVector3 rayAabbMaxLocal = rayFromLocal;
	rayAabbMaxLocal.setMax(rayToLocal);

	RecordingTriangleCallback rcb(rayFromLocal, rayToLocal, rayFlags);
	rcb.m_hitFraction = closestHitFraction;
	planeShape->processAllTriangles(&rcb, rayAabbMinLocal, rayAabbMaxLocal);

	g_PlaneVerificationStats.rays.fetch_add(1, std::memory_order_relaxed);

	if (rcb.m_hit != directHitFound)
	{
		g_PlaneVerificationStats.hitMismatches.fetch_add(1, std::memory_order_relaxed);
		btDumpPlaneMismatch(planeShape, colObjWorldTransform, rayFromWorld, rayToWorld, directHitFound);
		return;
	}
	if (!rcb.m_hit)
		return;

	g_PlaneVerificationStats.hits.fetch_add(1, std::memory_order_relaxed);

	// The fraction of the hit.
	const btScalar fractionDifference = btFabs(rcb.m_fraction - directHit.m_fraction);
	btAtomicMax(g_PlaneVerificationStats.maxFractionDifference, btDoubleToBits(fractionDifference));
	if (fractionDifference > btScalar(1e-5))
		g_PlaneVerificationStats.fractionMismatches.fetch_add(1, std::memory_order_relaxed);

	// The point of the hit, the way that the ray result callback computes it.
	btVector3 genericPoint, directPoint;
	genericPoint.setInterpolate3(rayFromWorld, rayToWorld, rcb.m_fraction);
	directPoint.setInterpolate3(rayFromWorld, rayToWorld, directHit.m_fraction);
	const btScalar pointDifference = (genericPoint - directPoint).length();
	btAtomicMax(g_PlaneVerificationStats.maxPointDifference, btDoubleToBits(pointDifference));
	if (pointDifference > btScalar(1e-4))
		g_PlaneVerificationStats.pointMismatches.fetch_add(1, std::memory_order_relaxed);

	// The normal of the hit.
	const btVector3 genericNormal = colObjWorldTransform.getBasis() * rcb.m_normalLocal;
	const btScalar normalDifference = (genericNormal - directHit.m_normalWorld).length();
	btAtomicMax(g_PlaneVerificationStats.maxNormalDifference, btDoubleToBits(normalDifference));
	if (normalDifference > btScalar(1e-4))
		g_PlaneVerificationStats.normalMismatches.fetch_add(1, std::memory_order_relaxed);

	// The metadata of the hit. The generic path decides the triangle with an edge test
	// that subtracts the coordinates of the projected center of the square, which are as
	// large as the arena is. Its answer is therefore not stable for a hit that is close
	// to the diagonal of the square: the same ray gives a different triangle when the
	// center moves by one unit in the last place. Such a hit is not compared, because the
	// direct ray test is the more precise one of the two.
	if (rcb.m_partId != 0 || rcb.m_triangleIndex != directHit.m_triangleIndex)
	{
		// The projected center is moved along the diagonal of the square, which is the
		// direction that the edge test of the generic path is the most sensitive to.
		btVector3 tangentDir0, tangentDir1;
		btPlaneSpace1(planeShape->getPlaneNormal(), tangentDir0, tangentDir1);
		const btVector3 diagonal = tangentDir0 - tangentDir1;

		btScalar magnitude = btScalar(0.0);
		for (int i = 0; i < 3; i++)
		{
			const btScalar center = btFabs((rayAabbMinLocal[i] + rayAabbMaxLocal[i]) * btScalar(0.5));
			magnitude = btMax(magnitude, center);
		}
		const btScalar ulp = std::nextafterf(magnitude, FLT_MAX) - magnitude;

		// The number of units in the last place of the coordinates at which the answer of
		// the generic path changes.
		int flipUlp = -1;
		for (int power = 0; power <= 6 && flipUlp < 0; power++)
		{
			const btScalar step = ulp * btScalar(1 << power);
			for (int sign = -1; sign <= 1 && flipUlp < 0; sign += 2)
			{
				const btVector3 shift = diagonal * (sign * step);

				RecordingTriangleCallback shiftedRcb(rayFromLocal, rayToLocal, rayFlags);
				shiftedRcb.m_hitFraction = closestHitFraction;
				planeShape->processAllTriangles(&shiftedRcb, rayAabbMinLocal + shift, rayAabbMaxLocal + shift);

				if (shiftedRcb.m_hit != rcb.m_hit ||
					shiftedRcb.m_triangleIndex != rcb.m_triangleIndex ||
					shiftedRcb.m_partId != rcb.m_partId)
				{
					flipUlp = 1 << power;
				}
			}
		}
		const bool stable = flipUlp < 0;

		if (stable)
		{
			g_PlaneVerificationStats.metadataMismatches.fetch_add(1, std::memory_order_relaxed);
			btAtomicMax(g_PlaneVerificationStats.maxMetadataMismatchOffset,
						 btDoubleToBits(btFabs(directHit.m_diagonalDistance + btScalar(0.0002) * directHit.m_radius) /
										directHit.m_radius));
		}
		else
		{
			g_PlaneVerificationStats.unstableMetadata.fetch_add(1, std::memory_order_relaxed);
		}
	}
}

void btCollisionWorld::rayTestSingleInternal(const btTransform& rayFromTrans, const btTransform& rayToTrans,
											 const btCollisionObjectWrapper* collisionObjectWrap,
											 RayResultCallback& resultCallback)
{
	RayShapeStatsTimer shapeTimer;
	if (btRayStatsEnabled())
	{
		int shapeType = collisionObjectWrap->getCollisionShape()->getShapeType();
		if (shapeType >= 0 && shapeType < 32)
		{
			g_RayShapeStats.counts[shapeType].fetch_add(1, std::memory_order_relaxed);
			shapeTimer.start = (uint64_t)std::chrono::steady_clock::now().time_since_epoch().count();
			shapeTimer.shapeType = shapeType;
		}
	}
	const btCollisionShape* collisionShape = collisionObjectWrap->getCollisionShape();
	const btTransform& colObjWorldTransform = collisionObjectWrap->getWorldTransform();

	if (collisionShape->isConvex())
	{
		//		BT_PROFILE("rayTestConvex");
		btSphereShape pointShape(btScalar(0.0));
		pointShape.setMargin(0.f);
		const btConvexShape* castShape = &pointShape;
		btConvexCast::CastResult castResult;
		castResult.m_fraction = resultCallback.m_closestHitFraction;

		btConvexShape* convexShape = (btConvexShape*)collisionShape;
		btVoronoiSimplexSolver simplexSolver;
		btSubsimplexConvexCast subSimplexConvexCaster(castShape, convexShape, &simplexSolver);

		if (subSimplexConvexCaster.calcTimeOfImpact(rayFromTrans, rayToTrans, colObjWorldTransform, colObjWorldTransform, castResult))
		{
			//add hit
			if (castResult.m_normal.length2() > btScalar(0.0001))
			{
				if (castResult.m_fraction < resultCallback.m_closestHitFraction)
				{
					//todo: figure out what this is about. When is rayFromTest.getBasis() not identity?
#ifdef USE_SUBSIMPLEX_CONVEX_CAST
					//rotate normal into worldspace
					castResult.m_normal = rayFromTrans.getBasis() * castResult.m_normal;
#endif  //USE_SUBSIMPLEX_CONVEX_CAST

					castResult.m_normal.normalize();
					btCollisionWorld::LocalRayResult localRayResult(
						collisionObjectWrap->getCollisionObject(),
						0,
						castResult.m_normal,
						castResult.m_fraction);

					bool normalInWorldSpace = true;
					resultCallback.addSingleResult(localRayResult, normalInWorldSpace);
				}
			}
		}
	}
	else
	{
		if (collisionShape->isConcave())
		{
			const int collisionShapeType = collisionShape->getShapeType();
			if (collisionShapeType == STATIC_PLANE_PROXYTYPE && btRayDirectPlaneEnabled())
			{
				// The direct ray test of the plane. It gives the same hit as the generic path
				// of the triangles below, but it does not build them.
				const btStaticPlaneShape* planeShape = (const btStaticPlaneShape*)collisionShape;
				const btVector3& rayFromWorld = rayFromTrans.getOrigin();
				const btVector3& rayToWorld = rayToTrans.getOrigin();

				btStaticPlaneHit directHit;
				const bool directHitFound = btRayTestStaticPlane(planeShape,
																colObjWorldTransform,
																rayFromWorld,
																rayToWorld,
																resultCallback.m_flags,
																resultCallback.m_closestHitFraction,
																directHit);

				if (btRayVerifyPlaneEnabled())
				{
					btVerifyStaticPlaneRay(planeShape,
										   colObjWorldTransform,
										   rayFromWorld,
										   rayToWorld,
										   resultCallback.m_flags,
										   resultCallback.m_closestHitFraction,
										   directHitFound,
										   directHit);
				}

				if (directHitFound)
				{
					btCollisionWorld::LocalShapeInfo shapeInfo;
					shapeInfo.m_shapePart = 0;
					shapeInfo.m_triangleIndex = directHit.m_triangleIndex;

					btCollisionWorld::LocalRayResult rayResult(collisionObjectWrap->getCollisionObject(),
															  &shapeInfo,
															  directHit.m_normalWorld,
															  directHit.m_fraction);

					const bool normalInWorldSpace = true;
					resultCallback.addSingleResult(rayResult, normalInWorldSpace);
				}
				return;
			}

			//ConvexCast::CastResult
			struct BridgeTriangleRaycastCallback : public btTriangleRaycastCallback
			{
				btCollisionWorld::RayResultCallback* m_resultCallback;
				const btCollisionObject* m_collisionObject;
				const btConcaveShape* m_triangleMesh;

				btTransform m_colObjWorldTransform;

				BridgeTriangleRaycastCallback(const btVector3& from, const btVector3& to,
											  btCollisionWorld::RayResultCallback* resultCallback, const btCollisionObject* collisionObject, const btConcaveShape* triangleMesh, const btTransform& colObjWorldTransform) :  //@BP Mod
																																																							btTriangleRaycastCallback(from, to, resultCallback->m_flags),
																																																							m_resultCallback(resultCallback),
																																																							m_collisionObject(collisionObject),
																																																							m_triangleMesh(triangleMesh),
																																																							m_colObjWorldTransform(colObjWorldTransform)
				{
				}

				virtual btScalar reportHit(const btVector3& hitNormalLocal, btScalar hitFraction, int partId, int triangleIndex)
				{
					btCollisionWorld::LocalShapeInfo shapeInfo;
					shapeInfo.m_shapePart = partId;
					shapeInfo.m_triangleIndex = triangleIndex;

					btVector3 hitNormalWorld = m_colObjWorldTransform.getBasis() * hitNormalLocal;

					btCollisionWorld::LocalRayResult rayResult(m_collisionObject,
															   &shapeInfo,
															   hitNormalWorld,
															   hitFraction);

					bool normalInWorldSpace = true;
					return m_resultCallback->addSingleResult(rayResult, normalInWorldSpace);
				}
			};

			btTransform worldTocollisionObject = colObjWorldTransform.inverse();
			btVector3 rayFromLocal = worldTocollisionObject * rayFromTrans.getOrigin();
			btVector3 rayToLocal = worldTocollisionObject * rayToTrans.getOrigin();

			//			BT_PROFILE("rayTestConcave");
			if (collisionShapeType == TRIANGLE_MESH_SHAPE_PROXYTYPE)
			{
				///optimized version for btBvhTriangleMeshShape
				btBvhTriangleMeshShape* triangleMesh = (btBvhTriangleMeshShape*)collisionShape;

				BridgeTriangleRaycastCallback rcb(rayFromLocal, rayToLocal, &resultCallback, collisionObjectWrap->getCollisionObject(), triangleMesh, colObjWorldTransform);
				rcb.m_hitFraction = resultCallback.m_closestHitFraction;
				triangleMesh->performRaycast(&rcb, rayFromLocal, rayToLocal);
			}
			else if (collisionShapeType == SCALED_TRIANGLE_MESH_SHAPE_PROXYTYPE)
			{
				btAssert(false);
			}
			else if (((resultCallback.m_flags&btTriangleRaycastCallback::kF_DisableHeightfieldAccelerator)==0)
				&& collisionShapeType == TERRAIN_SHAPE_PROXYTYPE
				)
			{
				btAssert(false);
			}
			else
			{
				//generic (slower) case
				btConcaveShape* concaveShape = (btConcaveShape*)collisionShape;

				btTransform worldTocollisionObject = colObjWorldTransform.inverse();

				btVector3 rayFromLocal = worldTocollisionObject * rayFromTrans.getOrigin();
				btVector3 rayToLocal = worldTocollisionObject * rayToTrans.getOrigin();

				//ConvexCast::CastResult

				struct BridgeTriangleRaycastCallback : public btTriangleRaycastCallback
				{
					btCollisionWorld::RayResultCallback* m_resultCallback;
					const btCollisionObject* m_collisionObject;
					btConcaveShape* m_triangleMesh;

					btTransform m_colObjWorldTransform;

					BridgeTriangleRaycastCallback(const btVector3& from, const btVector3& to,
												  btCollisionWorld::RayResultCallback* resultCallback, const btCollisionObject* collisionObject, btConcaveShape* triangleMesh, const btTransform& colObjWorldTransform) :  //@BP Mod
																																																						  btTriangleRaycastCallback(from, to, resultCallback->m_flags),
																																																						  m_resultCallback(resultCallback),
																																																						  m_collisionObject(collisionObject),
																																																						  m_triangleMesh(triangleMesh),
																																																						  m_colObjWorldTransform(colObjWorldTransform)
					{
					}

					virtual btScalar reportHit(const btVector3& hitNormalLocal, btScalar hitFraction, int partId, int triangleIndex)
					{
						btCollisionWorld::LocalShapeInfo shapeInfo;
						shapeInfo.m_shapePart = partId;
						shapeInfo.m_triangleIndex = triangleIndex;

						btVector3 hitNormalWorld = m_colObjWorldTransform.getBasis() * hitNormalLocal;

						btCollisionWorld::LocalRayResult rayResult(m_collisionObject,
																   &shapeInfo,
																   hitNormalWorld,
																   hitFraction);

						bool normalInWorldSpace = true;
						return m_resultCallback->addSingleResult(rayResult, normalInWorldSpace);
					}
				};

				BridgeTriangleRaycastCallback rcb(rayFromLocal, rayToLocal, &resultCallback, collisionObjectWrap->getCollisionObject(), concaveShape, colObjWorldTransform);
				rcb.m_hitFraction = resultCallback.m_closestHitFraction;

				btVector3 rayAabbMinLocal = rayFromLocal;
				rayAabbMinLocal.setMin(rayToLocal);
				btVector3 rayAabbMaxLocal = rayFromLocal;
				rayAabbMaxLocal.setMax(rayToLocal);

				concaveShape->processAllTriangles(&rcb, rayAabbMinLocal, rayAabbMaxLocal);
			}
		}
		else
		{
			//			BT_PROFILE("rayTestCompound");
			if (collisionShape->isCompound())
			{
				struct LocalInfoAdder2 : public RayResultCallback
				{
					RayResultCallback* m_userCallback;
					int m_i;

					LocalInfoAdder2(int i, RayResultCallback* user)
						: m_userCallback(user), m_i(i)
					{
						m_closestHitFraction = m_userCallback->m_closestHitFraction;
						m_flags = m_userCallback->m_flags;
					}
					virtual bool needsCollision(btBroadphaseProxy* p) const
					{
						return m_userCallback->needsCollision(p);
					}

					virtual btScalar addSingleResult(btCollisionWorld::LocalRayResult& r, bool b)
					{
						btCollisionWorld::LocalShapeInfo shapeInfo;
						shapeInfo.m_shapePart = -1;
						shapeInfo.m_triangleIndex = m_i;
						if (r.m_localShapeInfo == NULL)
							r.m_localShapeInfo = &shapeInfo;

						const btScalar result = m_userCallback->addSingleResult(r, b);
						m_closestHitFraction = m_userCallback->m_closestHitFraction;
						return result;
					}
				};

				struct RayTester : btDbvt::ICollide
				{
					const btCollisionObject* m_collisionObject;
					const btCompoundShape* m_compoundShape;
					const btTransform& m_colObjWorldTransform;
					const btTransform& m_rayFromTrans;
					const btTransform& m_rayToTrans;
					RayResultCallback& m_resultCallback;

					RayTester(const btCollisionObject* collisionObject,
							  const btCompoundShape* compoundShape,
							  const btTransform& colObjWorldTransform,
							  const btTransform& rayFromTrans,
							  const btTransform& rayToTrans,
							  RayResultCallback& resultCallback) : m_collisionObject(collisionObject),
																   m_compoundShape(compoundShape),
																   m_colObjWorldTransform(colObjWorldTransform),
																   m_rayFromTrans(rayFromTrans),
																   m_rayToTrans(rayToTrans),
																   m_resultCallback(resultCallback)
					{
					}

					void ProcessLeaf(int i)
					{
						const btCollisionShape* childCollisionShape = m_compoundShape->getChildShape(i);
						const btTransform& childTrans = m_compoundShape->getChildTransform(i);
						btTransform childWorldTrans = m_colObjWorldTransform * childTrans;

						btCollisionObjectWrapper tmpOb(0, childCollisionShape, m_collisionObject, childWorldTrans, -1, i);
						// replace collision shape so that callback can determine the triangle

						LocalInfoAdder2 my_cb(i, &m_resultCallback);

						rayTestSingleInternal(
							m_rayFromTrans,
							m_rayToTrans,
							&tmpOb,
							my_cb);
					}

					void Process(const btDbvtNode* leaf)
					{
						ProcessLeaf(leaf->dataAsInt);
					}
				};

				const btCompoundShape* compoundShape = static_cast<const btCompoundShape*>(collisionShape);
				const btDbvt* dbvt = compoundShape->getDynamicAabbTree();

				RayTester rayCB(
					collisionObjectWrap->getCollisionObject(),
					compoundShape,
					colObjWorldTransform,
					rayFromTrans,
					rayToTrans,
					resultCallback);
#ifndef DISABLE_DBVT_COMPOUNDSHAPE_RAYCAST_ACCELERATION
				if (dbvt)
				{
					btVector3 localRayFrom = colObjWorldTransform.inverseTimes(rayFromTrans).getOrigin();
					btVector3 localRayTo = colObjWorldTransform.inverseTimes(rayToTrans).getOrigin();
					btDbvt::rayTest(dbvt->m_root, localRayFrom, localRayTo, rayCB);
				}
				else
#endif  //DISABLE_DBVT_COMPOUNDSHAPE_RAYCAST_ACCELERATION
				{
					for (int i = 0, n = compoundShape->getNumChildShapes(); i < n; ++i)
					{
						rayCB.ProcessLeaf(i);
					}
				}
			}
		}
	}
}

void btCollisionWorld::objectQuerySingle(const btConvexShape* castShape, const btTransform& convexFromTrans, const btTransform& convexToTrans,
										 btCollisionObject* collisionObject,
										 const btCollisionShape* collisionShape,
										 const btTransform& colObjWorldTransform,
										 ConvexResultCallback& resultCallback, btScalar allowedPenetration)
{
	btCollisionObjectWrapper tmpOb(0, collisionShape, collisionObject, colObjWorldTransform, -1, -1);
	btCollisionWorld::objectQuerySingleInternal(castShape, convexFromTrans, convexToTrans, &tmpOb, resultCallback, allowedPenetration);
}

void btCollisionWorld::objectQuerySingleInternal(const btConvexShape* castShape, const btTransform& convexFromTrans, const btTransform& convexToTrans,
												 const btCollisionObjectWrapper* colObjWrap,
												 ConvexResultCallback& resultCallback, btScalar allowedPenetration)
{
	const btCollisionShape* collisionShape = colObjWrap->getCollisionShape();
	const btTransform& colObjWorldTransform = colObjWrap->getWorldTransform();

	if (collisionShape->isConvex())
	{
		//BT_PROFILE("convexSweepConvex");
		btConvexCast::CastResult castResult;
		castResult.m_allowedPenetration = allowedPenetration;
		castResult.m_fraction = resultCallback.m_closestHitFraction;  //btScalar(1.);//??

		btConvexShape* convexShape = (btConvexShape*)collisionShape;
		btVoronoiSimplexSolver simplexSolver;
		btGjkEpaPenetrationDepthSolver gjkEpaPenetrationSolver;

		btContinuousConvexCollision convexCaster1(castShape, convexShape, &simplexSolver, &gjkEpaPenetrationSolver);
		//btGjkConvexCast convexCaster2(castShape,convexShape,&simplexSolver);
		//btSubsimplexConvexCast convexCaster3(castShape,convexShape,&simplexSolver);

		btConvexCast* castPtr = &convexCaster1;

		if (castPtr->calcTimeOfImpact(convexFromTrans, convexToTrans, colObjWorldTransform, colObjWorldTransform, castResult))
		{
			//add hit
			if (castResult.m_normal.length2() > btScalar(0.0001))
			{
				if (castResult.m_fraction < resultCallback.m_closestHitFraction)
				{
					castResult.m_normal.normalize();
					btCollisionWorld::LocalConvexResult localConvexResult(
						colObjWrap->getCollisionObject(),
						0,
						castResult.m_normal,
						castResult.m_hitPoint,
						castResult.m_fraction);

					bool normalInWorldSpace = true;
					resultCallback.addSingleResult(localConvexResult, normalInWorldSpace);
				}
			}
		}
	}
	else
	{
		if (collisionShape->isConcave())
		{
			if (collisionShape->getShapeType() == TRIANGLE_MESH_SHAPE_PROXYTYPE)
			{
				//BT_PROFILE("convexSweepbtBvhTriangleMesh");
				btBvhTriangleMeshShape* triangleMesh = (btBvhTriangleMeshShape*)collisionShape;
				btTransform worldTocollisionObject = colObjWorldTransform.inverse();
				btVector3 convexFromLocal = worldTocollisionObject * convexFromTrans.getOrigin();
				btVector3 convexToLocal = worldTocollisionObject * convexToTrans.getOrigin();
				// rotation of box in local mesh space = MeshRotation^-1 * ConvexToRotation
				btTransform rotationXform = btTransform(worldTocollisionObject.getBasis() * convexToTrans.getBasis());

				//ConvexCast::CastResult
				struct BridgeTriangleConvexcastCallback : public btTriangleConvexcastCallback
				{
					btCollisionWorld::ConvexResultCallback* m_resultCallback;
					const btCollisionObject* m_collisionObject;
					btTriangleMeshShape* m_triangleMesh;

					BridgeTriangleConvexcastCallback(const btConvexShape* castShape, const btTransform& from, const btTransform& to,
													 btCollisionWorld::ConvexResultCallback* resultCallback, const btCollisionObject* collisionObject, btTriangleMeshShape* triangleMesh, const btTransform& triangleToWorld) : btTriangleConvexcastCallback(castShape, from, to, triangleToWorld, triangleMesh->getMargin()),
																																																								m_resultCallback(resultCallback),
																																																								m_collisionObject(collisionObject),
																																																								m_triangleMesh(triangleMesh)
					{
					}

					virtual btScalar reportHit(const btVector3& hitNormalLocal, const btVector3& hitPointLocal, btScalar hitFraction, int partId, int triangleIndex)
					{
						btCollisionWorld::LocalShapeInfo shapeInfo;
						shapeInfo.m_shapePart = partId;
						shapeInfo.m_triangleIndex = triangleIndex;
						if (hitFraction <= m_resultCallback->m_closestHitFraction)
						{
							btCollisionWorld::LocalConvexResult convexResult(m_collisionObject,
																			 &shapeInfo,
																			 hitNormalLocal,
																			 hitPointLocal,
																			 hitFraction);

							bool normalInWorldSpace = true;

							return m_resultCallback->addSingleResult(convexResult, normalInWorldSpace);
						}
						return hitFraction;
					}
				};

				BridgeTriangleConvexcastCallback tccb(castShape, convexFromTrans, convexToTrans, &resultCallback, colObjWrap->getCollisionObject(), triangleMesh, colObjWorldTransform);
				tccb.m_hitFraction = resultCallback.m_closestHitFraction;
				tccb.m_allowedPenetration = allowedPenetration;
				btVector3 boxMinLocal, boxMaxLocal;
				castShape->getAabb(rotationXform, boxMinLocal, boxMaxLocal);
				triangleMesh->performConvexcast(&tccb, convexFromLocal, convexToLocal, boxMinLocal, boxMaxLocal);
			}
			else
			{
				if (collisionShape->getShapeType() == STATIC_PLANE_PROXYTYPE)
				{
					btConvexCast::CastResult castResult;
					castResult.m_allowedPenetration = allowedPenetration;
					castResult.m_fraction = resultCallback.m_closestHitFraction;
					btStaticPlaneShape* planeShape = (btStaticPlaneShape*)collisionShape;
					btContinuousConvexCollision convexCaster1(castShape, planeShape);
					btConvexCast* castPtr = &convexCaster1;

					if (castPtr->calcTimeOfImpact(convexFromTrans, convexToTrans, colObjWorldTransform, colObjWorldTransform, castResult))
					{
						//add hit
						if (castResult.m_normal.length2() > btScalar(0.0001))
						{
							if (castResult.m_fraction < resultCallback.m_closestHitFraction)
							{
								castResult.m_normal.normalize();
								btCollisionWorld::LocalConvexResult localConvexResult(
									colObjWrap->getCollisionObject(),
									0,
									castResult.m_normal,
									castResult.m_hitPoint,
									castResult.m_fraction);

								bool normalInWorldSpace = true;
								resultCallback.addSingleResult(localConvexResult, normalInWorldSpace);
							}
						}
					}
				}
				else
				{
					//BT_PROFILE("convexSweepConcave");
					btConcaveShape* concaveShape = (btConcaveShape*)collisionShape;
					btTransform worldTocollisionObject = colObjWorldTransform.inverse();
					btVector3 convexFromLocal = worldTocollisionObject * convexFromTrans.getOrigin();
					btVector3 convexToLocal = worldTocollisionObject * convexToTrans.getOrigin();
					// rotation of box in local mesh space = MeshRotation^-1 * ConvexToRotation
					btTransform rotationXform = btTransform(worldTocollisionObject.getBasis() * convexToTrans.getBasis());

					//ConvexCast::CastResult
					struct BridgeTriangleConvexcastCallback : public btTriangleConvexcastCallback
					{
						btCollisionWorld::ConvexResultCallback* m_resultCallback;
						const btCollisionObject* m_collisionObject;
						btConcaveShape* m_triangleMesh;

						BridgeTriangleConvexcastCallback(const btConvexShape* castShape, const btTransform& from, const btTransform& to,
														 btCollisionWorld::ConvexResultCallback* resultCallback, const btCollisionObject* collisionObject, btConcaveShape* triangleMesh, const btTransform& triangleToWorld) : btTriangleConvexcastCallback(castShape, from, to, triangleToWorld, triangleMesh->getMargin()),
																																																							   m_resultCallback(resultCallback),
																																																							   m_collisionObject(collisionObject),
																																																							   m_triangleMesh(triangleMesh)
						{
						}

						virtual btScalar reportHit(const btVector3& hitNormalLocal, const btVector3& hitPointLocal, btScalar hitFraction, int partId, int triangleIndex)
						{
							btCollisionWorld::LocalShapeInfo shapeInfo;
							shapeInfo.m_shapePart = partId;
							shapeInfo.m_triangleIndex = triangleIndex;
							if (hitFraction <= m_resultCallback->m_closestHitFraction)
							{
								btCollisionWorld::LocalConvexResult convexResult(m_collisionObject,
																				 &shapeInfo,
																				 hitNormalLocal,
																				 hitPointLocal,
																				 hitFraction);

								bool normalInWorldSpace = true;

								return m_resultCallback->addSingleResult(convexResult, normalInWorldSpace);
							}
							return hitFraction;
						}
					};

					BridgeTriangleConvexcastCallback tccb(castShape, convexFromTrans, convexToTrans, &resultCallback, colObjWrap->getCollisionObject(), concaveShape, colObjWorldTransform);
					tccb.m_hitFraction = resultCallback.m_closestHitFraction;
					tccb.m_allowedPenetration = allowedPenetration;
					btVector3 boxMinLocal, boxMaxLocal;
					castShape->getAabb(rotationXform, boxMinLocal, boxMaxLocal);

					btVector3 rayAabbMinLocal = convexFromLocal;
					rayAabbMinLocal.setMin(convexToLocal);
					btVector3 rayAabbMaxLocal = convexFromLocal;
					rayAabbMaxLocal.setMax(convexToLocal);
					rayAabbMinLocal += boxMinLocal;
					rayAabbMaxLocal += boxMaxLocal;
					concaveShape->processAllTriangles(&tccb, rayAabbMinLocal, rayAabbMaxLocal);
				}
			}
		}
		else
		{
			if (collisionShape->isCompound())
			{
				struct btCompoundLeafCallback : btDbvt::ICollide
				{
					btCompoundLeafCallback(
						const btCollisionObjectWrapper* colObjWrap,
						const btConvexShape* castShape,
						const btTransform& convexFromTrans,
						const btTransform& convexToTrans,
						btScalar allowedPenetration,
						const btCompoundShape* compoundShape,
						const btTransform& colObjWorldTransform,
						ConvexResultCallback& resultCallback)
						: m_colObjWrap(colObjWrap),
						  m_castShape(castShape),
						  m_convexFromTrans(convexFromTrans),
						  m_convexToTrans(convexToTrans),
						  m_allowedPenetration(allowedPenetration),
						  m_compoundShape(compoundShape),
						  m_colObjWorldTransform(colObjWorldTransform),
						  m_resultCallback(resultCallback)
					{
					}

					const btCollisionObjectWrapper* m_colObjWrap;
					const btConvexShape* m_castShape;
					const btTransform& m_convexFromTrans;
					const btTransform& m_convexToTrans;
					btScalar m_allowedPenetration;
					const btCompoundShape* m_compoundShape;
					const btTransform& m_colObjWorldTransform;
					ConvexResultCallback& m_resultCallback;

				public:
					void ProcessChild(int index, const btTransform& childTrans, const btCollisionShape* childCollisionShape)
					{
						btTransform childWorldTrans = m_colObjWorldTransform * childTrans;

						struct LocalInfoAdder : public ConvexResultCallback
						{
							ConvexResultCallback* m_userCallback;
							int m_i;

							LocalInfoAdder(int i, ConvexResultCallback* user)
								: m_userCallback(user), m_i(i)
							{
								m_closestHitFraction = m_userCallback->m_closestHitFraction;
							}
							virtual bool needsCollision(btBroadphaseProxy* p) const
							{
								return m_userCallback->needsCollision(p);
							}
							virtual btScalar addSingleResult(btCollisionWorld::LocalConvexResult& r, bool b)
							{
								btCollisionWorld::LocalShapeInfo shapeInfo;
								shapeInfo.m_shapePart = -1;
								shapeInfo.m_triangleIndex = m_i;
								if (r.m_localShapeInfo == NULL)
									r.m_localShapeInfo = &shapeInfo;
								const btScalar result = m_userCallback->addSingleResult(r, b);
								m_closestHitFraction = m_userCallback->m_closestHitFraction;
								return result;
							}
						};

						LocalInfoAdder my_cb(index, &m_resultCallback);

						btCollisionObjectWrapper tmpObj(m_colObjWrap, childCollisionShape, m_colObjWrap->getCollisionObject(), childWorldTrans, -1, index);

						objectQuerySingleInternal(m_castShape, m_convexFromTrans, m_convexToTrans, &tmpObj, my_cb, m_allowedPenetration);
					}

					void Process(const btDbvtNode* leaf)
					{
						// Processing leaf node
						int index = leaf->dataAsInt;

						btTransform childTrans = m_compoundShape->getChildTransform(index);
						const btCollisionShape* childCollisionShape = m_compoundShape->getChildShape(index);

						ProcessChild(index, childTrans, childCollisionShape);
					}
				};

				BT_PROFILE("convexSweepCompound");
				const btCompoundShape* compoundShape = static_cast<const btCompoundShape*>(collisionShape);

				btVector3 fromLocalAabbMin, fromLocalAabbMax;
				btVector3 toLocalAabbMin, toLocalAabbMax;

				castShape->getAabb(colObjWorldTransform.inverse() * convexFromTrans, fromLocalAabbMin, fromLocalAabbMax);
				castShape->getAabb(colObjWorldTransform.inverse() * convexToTrans, toLocalAabbMin, toLocalAabbMax);

				fromLocalAabbMin.setMin(toLocalAabbMin);
				fromLocalAabbMax.setMax(toLocalAabbMax);

				btCompoundLeafCallback callback(colObjWrap, castShape, convexFromTrans, convexToTrans,
												allowedPenetration, compoundShape, colObjWorldTransform, resultCallback);

				const btDbvt* tree = compoundShape->getDynamicAabbTree();
				if (tree)
				{
					const ATTRIBUTE_ALIGNED16(btDbvtVolume) bounds = btDbvtVolume::FromMM(fromLocalAabbMin, fromLocalAabbMax);
					tree->collideTV(tree->m_root, bounds, callback);
				}
				else
				{
					int i;
					for (i = 0; i < compoundShape->getNumChildShapes(); i++)
					{
						const btCollisionShape* childCollisionShape = compoundShape->getChildShape(i);
						btTransform childTrans = compoundShape->getChildTransform(i);
						callback.ProcessChild(i, childTrans, childCollisionShape);
					}
				}
			}
		}
	}
}

struct btSingleRayCallback : public btBroadphaseRayCallback
{
	btVector3 m_rayFromWorld;
	btVector3 m_rayToWorld;
	btTransform m_rayFromTrans;
	btTransform m_rayToTrans;
	btVector3 m_hitNormal;

	const btCollisionWorld* m_world;
	btCollisionWorld::RayResultCallback& m_resultCallback;

	btSingleRayCallback(const btVector3& rayFromWorld, const btVector3& rayToWorld, const btCollisionWorld* world, btCollisionWorld::RayResultCallback& resultCallback)
		: m_rayFromWorld(rayFromWorld),
		  m_rayToWorld(rayToWorld),
		  m_world(world),
		  m_resultCallback(resultCallback)
	{
		m_rayFromTrans.setIdentity();
		m_rayFromTrans.setOrigin(m_rayFromWorld);
		m_rayToTrans.setIdentity();
		m_rayToTrans.setOrigin(m_rayToWorld);

		btVector3 rayDir = (rayToWorld - rayFromWorld);

		rayDir.normalize();
		///what about division by zero? --> just set rayDirection[i] to INF/BT_LARGE_FLOAT
		m_rayDirectionInverse[0] = rayDir[0] == btScalar(0.0) ? btScalar(BT_LARGE_FLOAT) : btScalar(1.0) / rayDir[0];
		m_rayDirectionInverse[1] = rayDir[1] == btScalar(0.0) ? btScalar(BT_LARGE_FLOAT) : btScalar(1.0) / rayDir[1];
		m_rayDirectionInverse[2] = rayDir[2] == btScalar(0.0) ? btScalar(BT_LARGE_FLOAT) : btScalar(1.0) / rayDir[2];
		m_signs[0] = m_rayDirectionInverse[0] < 0.0;
		m_signs[1] = m_rayDirectionInverse[1] < 0.0;
		m_signs[2] = m_rayDirectionInverse[2] < 0.0;

		m_lambda_max = rayDir.dot(m_rayToWorld - m_rayFromWorld);
	}

	virtual bool process(const btBroadphaseProxy* proxy)
	{
		///terminate further ray tests, once the closestHitFraction reached zero
		if (m_resultCallback.m_closestHitFraction == btScalar(0.f))
			return false;

		btCollisionObject* collisionObject = (btCollisionObject*)proxy->m_clientObject;

		//only perform raycast if filterMask matches
		if (m_resultCallback.needsCollision(collisionObject->getBroadphaseHandle()))
		{
			//RigidcollisionObject* collisionObject = ctrl->GetRigidcollisionObject();
			//btVector3 collisionObjectAabbMin,collisionObjectAabbMax;
#if 0
#ifdef RECALCULATE_AABB
			btVector3 collisionObjectAabbMin,collisionObjectAabbMax;
			collisionObject->getCollisionShape()->getAabb(collisionObject->getWorldTransform(),collisionObjectAabbMin,collisionObjectAabbMax);
#else
			//getBroadphase()->getAabb(collisionObject->getBroadphaseHandle(),collisionObjectAabbMin,collisionObjectAabbMax);
			const btVector3& collisionObjectAabbMin = collisionObject->getBroadphaseHandle()->m_aabbMin;
			const btVector3& collisionObjectAabbMax = collisionObject->getBroadphaseHandle()->m_aabbMax;
#endif
#endif
			//btScalar hitLambda = m_resultCallback.m_closestHitFraction;
			//culling already done by broadphase
			//if (btRayAabb(m_rayFromWorld,m_rayToWorld,collisionObjectAabbMin,collisionObjectAabbMax,hitLambda,m_hitNormal))
			{
				m_world->rayTestSingle(m_rayFromTrans, m_rayToTrans,
									   collisionObject,
									   collisionObject->getCollisionShape(),
									   collisionObject->getWorldTransform(),
									   m_resultCallback);
			}
		}
		return true;
	}
};

void btCollisionWorld::rayTest(const btVector3& rayFromWorld, const btVector3& rayToWorld, RayResultCallback& resultCallback) const
{
	//BT_PROFILE("rayTest");
	/// use the broadphase to accelerate the search for objects, based on their aabb
	/// and for each object with ray-aabb overlap, perform an exact ray test
	btSingleRayCallback rayCB(rayFromWorld, rayToWorld, this, resultCallback);

#ifndef USE_BRUTEFORCE_RAYBROADPHASE
	m_broadphasePairCache->rayTest(rayFromWorld, rayToWorld, rayCB);
#else
	for (int i = 0; i < this->getNumCollisionObjects(); i++)
	{
		rayCB.process(m_collisionObjects[i]->getBroadphaseHandle());
	}
#endif  //USE_BRUTEFORCE_RAYBROADPHASE
}

struct btSingleSweepCallback : public btBroadphaseRayCallback
{
	btTransform m_convexFromTrans;
	btTransform m_convexToTrans;
	btVector3 m_hitNormal;
	const btCollisionWorld* m_world;
	btCollisionWorld::ConvexResultCallback& m_resultCallback;
	btScalar m_allowedCcdPenetration;
	const btConvexShape* m_castShape;

	btSingleSweepCallback(const btConvexShape* castShape, const btTransform& convexFromTrans, const btTransform& convexToTrans, const btCollisionWorld* world, btCollisionWorld::ConvexResultCallback& resultCallback, btScalar allowedPenetration)
		: m_convexFromTrans(convexFromTrans),
		  m_convexToTrans(convexToTrans),
		  m_world(world),
		  m_resultCallback(resultCallback),
		  m_allowedCcdPenetration(allowedPenetration),
		  m_castShape(castShape)
	{
		btVector3 unnormalizedRayDir = (m_convexToTrans.getOrigin() - m_convexFromTrans.getOrigin());
		btVector3 rayDir = unnormalizedRayDir.fuzzyZero() ? btVector3(btScalar(0.0), btScalar(0.0), btScalar(0.0)) : unnormalizedRayDir.normalized();
		///what about division by zero? --> just set rayDirection[i] to INF/BT_LARGE_FLOAT
		m_rayDirectionInverse[0] = rayDir[0] == btScalar(0.0) ? btScalar(BT_LARGE_FLOAT) : btScalar(1.0) / rayDir[0];
		m_rayDirectionInverse[1] = rayDir[1] == btScalar(0.0) ? btScalar(BT_LARGE_FLOAT) : btScalar(1.0) / rayDir[1];
		m_rayDirectionInverse[2] = rayDir[2] == btScalar(0.0) ? btScalar(BT_LARGE_FLOAT) : btScalar(1.0) / rayDir[2];
		m_signs[0] = m_rayDirectionInverse[0] < 0.0;
		m_signs[1] = m_rayDirectionInverse[1] < 0.0;
		m_signs[2] = m_rayDirectionInverse[2] < 0.0;

		m_lambda_max = rayDir.dot(unnormalizedRayDir);
	}

	virtual bool process(const btBroadphaseProxy* proxy)
	{
		///terminate further convex sweep tests, once the closestHitFraction reached zero
		if (m_resultCallback.m_closestHitFraction == btScalar(0.f))
			return false;

		btCollisionObject* collisionObject = (btCollisionObject*)proxy->m_clientObject;

		//only perform raycast if filterMask matches
		if (m_resultCallback.needsCollision(collisionObject->getBroadphaseHandle()))
		{
			//RigidcollisionObject* collisionObject = ctrl->GetRigidcollisionObject();
			m_world->objectQuerySingle(m_castShape, m_convexFromTrans, m_convexToTrans,
									   collisionObject,
									   collisionObject->getCollisionShape(),
									   collisionObject->getWorldTransform(),
									   m_resultCallback,
									   m_allowedCcdPenetration);
		}

		return true;
	}
};

void btCollisionWorld::convexSweepTest(const btConvexShape* castShape, const btTransform& convexFromWorld, const btTransform& convexToWorld, ConvexResultCallback& resultCallback, btScalar allowedCcdPenetration) const
{
	BT_PROFILE("convexSweepTest");
	/// use the broadphase to accelerate the search for objects, based on their aabb
	/// and for each object with ray-aabb overlap, perform an exact ray test
	/// unfortunately the implementation for rayTest and convexSweepTest duplicated, albeit practically identical

	btTransform convexFromTrans, convexToTrans;
	convexFromTrans = convexFromWorld;
	convexToTrans = convexToWorld;
	btVector3 castShapeAabbMin, castShapeAabbMax;
	/* Compute AABB that encompasses angular movement */
	{
		btVector3 linVel, angVel;
		btTransformUtil::calculateVelocity(convexFromTrans, convexToTrans, 1.0f, linVel, angVel);
		btVector3 zeroLinVel;
		zeroLinVel.setValue(0, 0, 0);
		btTransform R;
		R.setIdentity();
		R.setRotation(convexFromTrans.getRotation());
		castShape->calculateTemporalAabb(R, zeroLinVel, angVel, 1.0f, castShapeAabbMin, castShapeAabbMax);
	}

#ifndef USE_BRUTEFORCE_RAYBROADPHASE

	btSingleSweepCallback convexCB(castShape, convexFromWorld, convexToWorld, this, resultCallback, allowedCcdPenetration);

	m_broadphasePairCache->rayTest(convexFromTrans.getOrigin(), convexToTrans.getOrigin(), convexCB, castShapeAabbMin, castShapeAabbMax);

#else
	/// go over all objects, and if the ray intersects their aabb + cast shape aabb,
	// do a ray-shape query using convexCaster (CCD)
	int i;
	for (i = 0; i < m_collisionObjects.size(); i++)
	{
		btCollisionObject* collisionObject = m_collisionObjects[i];
		//only perform raycast if filterMask matches
		if (resultCallback.needsCollision(collisionObject->getBroadphaseHandle()))
		{
			//RigidcollisionObject* collisionObject = ctrl->GetRigidcollisionObject();
			btVector3 collisionObjectAabbMin, collisionObjectAabbMax;
			collisionObject->getCollisionShape()->getAabb(collisionObject->getWorldTransform(), collisionObjectAabbMin, collisionObjectAabbMax);
			AabbExpand(collisionObjectAabbMin, collisionObjectAabbMax, castShapeAabbMin, castShapeAabbMax);
			btScalar hitLambda = btScalar(1.);  //could use resultCallback.m_closestHitFraction, but needs testing
			btVector3 hitNormal;
			if (btRayAabb(convexFromWorld.getOrigin(), convexToWorld.getOrigin(), collisionObjectAabbMin, collisionObjectAabbMax, hitLambda, hitNormal))
			{
				objectQuerySingle(castShape, convexFromTrans, convexToTrans,
								  collisionObject,
								  collisionObject->getCollisionShape(),
								  collisionObject->getWorldTransform(),
								  resultCallback,
								  allowedCcdPenetration);
			}
		}
	}
#endif  //USE_BRUTEFORCE_RAYBROADPHASE
}

struct btBridgedManifoldResult : public btManifoldResult
{
	btCollisionWorld::ContactResultCallback& m_resultCallback;

	btBridgedManifoldResult(const btCollisionObjectWrapper* obj0Wrap, const btCollisionObjectWrapper* obj1Wrap, btCollisionWorld::ContactResultCallback& resultCallback)
		: btManifoldResult(obj0Wrap, obj1Wrap),
		  m_resultCallback(resultCallback)
	{
	}

	virtual void addContactPoint(const btVector3& normalOnBInWorld, const btVector3& pointInWorld, btScalar depth)
	{
		bool isSwapped = m_manifoldPtr->getBody0() != m_body0Wrap->getCollisionObject();
		btVector3 pointA = pointInWorld + normalOnBInWorld * depth;
		btVector3 localA;
		btVector3 localB;
		if (isSwapped)
		{
			localA = m_body1Wrap->getCollisionObject()->getWorldTransform().invXform(pointA);
			localB = m_body0Wrap->getCollisionObject()->getWorldTransform().invXform(pointInWorld);
		}
		else
		{
			localA = m_body0Wrap->getCollisionObject()->getWorldTransform().invXform(pointA);
			localB = m_body1Wrap->getCollisionObject()->getWorldTransform().invXform(pointInWorld);
		}

		btManifoldPoint newPt(localA, localB, normalOnBInWorld, depth);
		newPt.m_positionWorldOnA = pointA;
		newPt.m_positionWorldOnB = pointInWorld;

		//BP mod, store contact triangles.
		if (isSwapped)
		{
			newPt.m_partId0 = m_partId1;
			newPt.m_partId1 = m_partId0;
			newPt.m_index0 = m_index1;
			newPt.m_index1 = m_index0;
		}
		else
		{
			newPt.m_partId0 = m_partId0;
			newPt.m_partId1 = m_partId1;
			newPt.m_index0 = m_index0;
			newPt.m_index1 = m_index1;
		}

		//experimental feature info, for per-triangle material etc.
		const btCollisionObjectWrapper* obj0Wrap = isSwapped ? m_body1Wrap : m_body0Wrap;
		const btCollisionObjectWrapper* obj1Wrap = isSwapped ? m_body0Wrap : m_body1Wrap;
		m_resultCallback.addSingleResult(newPt, obj0Wrap, newPt.m_partId0, newPt.m_index0, obj1Wrap, newPt.m_partId1, newPt.m_index1);
	}
};

struct btSingleContactCallback : public btBroadphaseAabbCallback
{
	btCollisionObject* m_collisionObject;
	btCollisionWorld* m_world;
	btCollisionWorld::ContactResultCallback& m_resultCallback;

	btSingleContactCallback(btCollisionObject* collisionObject, btCollisionWorld* world, btCollisionWorld::ContactResultCallback& resultCallback)
		: m_collisionObject(collisionObject),
		  m_world(world),
		  m_resultCallback(resultCallback)
	{
	}

	virtual bool process(const btBroadphaseProxy* proxy)
	{
		btCollisionObject* collisionObject = (btCollisionObject*)proxy->m_clientObject;
		if (collisionObject == m_collisionObject)
			return true;

		//only perform raycast if filterMask matches
		if (m_resultCallback.needsCollision(collisionObject->getBroadphaseHandle()))
		{
			btCollisionObjectWrapper ob0(0, m_collisionObject->getCollisionShape(), m_collisionObject, m_collisionObject->getWorldTransform(), -1, -1);
			btCollisionObjectWrapper ob1(0, collisionObject->getCollisionShape(), collisionObject, collisionObject->getWorldTransform(), -1, -1);

			btCollisionAlgorithm* algorithm = m_world->getDispatcher()->findAlgorithm(&ob0, &ob1, 0, BT_CLOSEST_POINT_ALGORITHMS);
			if (algorithm)
			{
				btBridgedManifoldResult contactPointResult(&ob0, &ob1, m_resultCallback);
				//discrete collision detection query

				algorithm->processCollision(&ob0, &ob1, m_world->getDispatchInfo(), &contactPointResult);

				algorithm->~btCollisionAlgorithm();
				m_world->getDispatcher()->freeCollisionAlgorithm(algorithm);
			}
		}
		return true;
	}
};

///contactTest performs a discrete collision test against all objects in the btCollisionWorld, and calls the resultCallback.
///it reports one or more contact points for every overlapping object (including the one with deepest penetration)
void btCollisionWorld::contactTest(btCollisionObject* colObj, ContactResultCallback& resultCallback)
{
	btVector3 aabbMin, aabbMax;
	colObj->getCollisionShape()->getAabb(colObj->getWorldTransform(), aabbMin, aabbMax);
	btSingleContactCallback contactCB(colObj, this, resultCallback);

	m_broadphasePairCache->aabbTest(aabbMin, aabbMax, contactCB);
}

///contactTest performs a discrete collision test between two collision objects and calls the resultCallback if overlap if detected.
///it reports one or more contact points (including the one with deepest penetration)
void btCollisionWorld::contactPairTest(btCollisionObject* colObjA, btCollisionObject* colObjB, ContactResultCallback& resultCallback)
{
	btCollisionObjectWrapper obA(0, colObjA->getCollisionShape(), colObjA, colObjA->getWorldTransform(), -1, -1);
	btCollisionObjectWrapper obB(0, colObjB->getCollisionShape(), colObjB, colObjB->getWorldTransform(), -1, -1);

	btCollisionAlgorithm* algorithm = getDispatcher()->findAlgorithm(&obA, &obB, 0, BT_CLOSEST_POINT_ALGORITHMS);
	if (algorithm)
	{
		btBridgedManifoldResult contactPointResult(&obA, &obB, resultCallback);
		contactPointResult.m_closestPointDistanceThreshold = resultCallback.m_closestDistanceThreshold;
		//discrete collision detection query
		algorithm->processCollision(&obA, &obB, getDispatchInfo(), &contactPointResult);

		algorithm->~btCollisionAlgorithm();
		getDispatcher()->freeCollisionAlgorithm(algorithm);
	}
}

bool btCollisionWorld::RayResultCallback::needsCollision(btBroadphaseProxy* proxy0) const {
	bool collides = (proxy0->m_collisionFilterGroup & m_collisionFilterMask) != 0
		&& (m_collisionFilterGroup & proxy0->m_collisionFilterMask);
	if (proxy0->m_clientObject == m_ignoreObj)
		return false;
	return collides;
}