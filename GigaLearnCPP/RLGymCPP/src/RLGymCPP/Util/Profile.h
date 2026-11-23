#pragma once

// Lightweight phase timers for finding out where the training time goes.
// Off by default: set the GGL_PROFILE environment variable to 1 to collect them.
// The collected times are added to the learner's report (see Learner.cpp).

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>

namespace GGL {

	// Whether the profiler collects the timings. It is turned on with the GGL_PROFILE
	// environment variable. The counters do nothing when it is off, so that the production
	// run has no atomic operations in its hot paths.
	inline bool ProfileEnabled() {
		static const bool enabled = []() {
			const char* value = getenv("GGL_PROFILE");
			return value != NULL && value[0] != '\0' && value[0] != '0';
		}();
		return enabled;
	}

	struct ProfCounter {
		std::atomic<uint64_t> ns{ 0 };
		std::atomic<uint64_t> count{ 0 };
		std::atomic<uint64_t> flops{ 0 };

		void Add(uint64_t value) {
			if (!ProfileEnabled())
				return;
			ns.fetch_add(value, std::memory_order_relaxed);
			count.fetch_add(1, std::memory_order_relaxed);
		}

		void AddCount(uint64_t value = 1) {
			if (!ProfileEnabled())
				return;
			count.fetch_add(value, std::memory_order_relaxed);
		}

		void AddFlops(uint64_t value) {
			if (!ProfileEnabled())
				return;
			flops.fetch_add(value, std::memory_order_relaxed);
		}

		double Seconds() const {
			return (double)ns.load(std::memory_order_relaxed) * 1e-9;
		}

		double GFlops() const {
			return (double)flops.load(std::memory_order_relaxed) * 1e-9;
		}

		uint64_t Count() const {
			return count.load(std::memory_order_relaxed);
		}

		void Reset() {
			ns.store(0, std::memory_order_relaxed);
			count.store(0, std::memory_order_relaxed);
			flops.store(0, std::memory_order_relaxed);
		}
	};

	struct Profile {
		// The matrix products and the normalization kernels of the fused layers
		ProfCounter mmFwd, mmBwd, kernFwd, kernBwd;
		// The forward pass of every model (the shared head includes the slot encoder)
		ProfCounter fwdShared, fwdPolicy, fwdCritic;
		// The slot encoder's data movement
		ProfCounter slotGather, slotPool;
		// The phases of the PPO update
		ProfCounter ppoGather, ppoFwd, ppoBwd, ppoClip, ppoOptim, ppoCopy;
		// The phases of the collection
		ProfCounter idxSelect, inferNew, inferOld, obsBook, envSync, envSecondHalf;
		// The parts of the two halves of an environment step
		ProfCounter fhCopy, fhReset, fhStep, shParse, shStep, shGsUpdate, shTerminal, shRewards;
		// The matrix products of the inference (no gradient) and of the PPO update (with gradient)
		ProfCounter mmInfer, mmPpo, inferCalls, inferRows;
		// The matrix products of the three models (shared head, policy, critic)
		ProfCounter mmShared, mmPolicy, mmCritic, modelCalls;

		void Reset() {
			mmFwd.Reset(); mmBwd.Reset(); kernFwd.Reset(); kernBwd.Reset();
			fwdShared.Reset(); fwdPolicy.Reset(); fwdCritic.Reset();
			slotGather.Reset(); slotPool.Reset();
			ppoGather.Reset(); ppoFwd.Reset(); ppoBwd.Reset();
			ppoClip.Reset(); ppoOptim.Reset(); ppoCopy.Reset();
			idxSelect.Reset(); inferNew.Reset(); inferOld.Reset();
			obsBook.Reset(); envSync.Reset(); envSecondHalf.Reset();
			fhCopy.Reset(); fhReset.Reset(); fhStep.Reset();
			shParse.Reset(); shStep.Reset(); shGsUpdate.Reset(); shTerminal.Reset(); shRewards.Reset();
			mmInfer.Reset(); mmPpo.Reset(); inferCalls.Reset(); inferRows.Reset();
			mmShared.Reset(); mmPolicy.Reset(); mmCritic.Reset(); modelCalls.Reset();
		}
	};

	inline Profile g_Profile;

	// The matrix products of the current thread, so that the work of one model can be told
	// apart from the work of the other threads. The shared counters are only used for the
	// totals, because their differences would include the work of the other threads.
	inline thread_local uint64_t g_TlsMmFwdFlops = 0;

	// Turns the timings off for the thread that is set, without turning them off for the
	// others. It is used when two pieces of code are compared with each other, so that
	// neither of them is charged for the timers.
	inline thread_local bool g_TlsProfileOff = false;


	// Times the code between its construction and its destruction
	struct ProfScope {
		ProfCounter* counter = NULL;
		std::chrono::steady_clock::time_point start;

		ProfScope(ProfCounter& target) {
			if (!ProfileEnabled() || g_TlsProfileOff)
				return;
			counter = &target;
			start = std::chrono::steady_clock::now();
		}

		ProfScope(ProfCounter* target) {
			if (target == NULL || !ProfileEnabled() || g_TlsProfileOff)
				return;
			counter = target;
			start = std::chrono::steady_clock::now();
		}

		~ProfScope() {
			if (counter == NULL)
				return;
			uint64_t elapsed = (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
				std::chrono::steady_clock::now() - start).count();
			counter->Add(elapsed);
		}
	};

}
