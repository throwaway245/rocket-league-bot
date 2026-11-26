#include "InferUnit.h"

#include <GigaLearnCPP/Util/Models.h>
#include <GigaLearnCPP/PPO/PPOLearner.h>
#include <torch/serialize.h>

GGL::InferUnit::InferUnit(
	RLGC::ObsBuilder* obsBuilder, int obsSize, RLGC::ActionParser* actionParser,
	PartialModelConfig sharedHeadConfig, PartialModelConfig policyConfig, 
	std::filesystem::path modelsFolder, bool useGPU) : 
	obsBuilder(obsBuilder), obsSize(obsSize), actionParser(actionParser), useGPU(useGPU) {

	this->models = new ModelSet();

	try {
		PPOLearner::MakeModels(
			false, obsSize, actionParser->GetActionAmount(),
			sharedHeadConfig, policyConfig, {},
			useGPU ? torch::kCUDA : torch::kCPU,
			*this->models
		);
	} catch (std::exception& e) {
		RG_ERR_CLOSE("InferUnit: Exception when trying to construct models: " << e.what());
	}

	try {
		this->models->Load(modelsFolder, false, false);
	} catch (std::exception& e) {
		RG_ERR_CLOSE("InferUnit: Exception when trying to load models: " << e.what());
	}
}

void GGL::InferUnit::SetMirrorX(int mirrorX) {
	if (mirrorX && !obsBuilder->SupportsMirrorX())
		RG_ERR_CLOSE("InferUnit::SetMirrorX(): The obs builder does not support the mirrored view");
	obsBuilder->mirrorXTeam[0] = obsBuilder->mirrorXTeam[1] = mirrorX;
}

void GGL::InferUnit::SaveInferenceTrace(const InferenceTrace& trace, int obsSize, int actionCount, const std::filesystem::path& path) {
	TORCH_CHECK(!std::filesystem::exists(path), "Refusing to overwrite match replay");
	auto rows = (int64_t)trace.actions.size();
	TORCH_CHECK(rows > 0 && trace.observations.size() == rows * obsSize && trace.actionMasks.size() == rows * actionCount,
		"Invalid or empty match trace");
	auto obs = torch::tensor(trace.observations).reshape({rows, obsSize});
	auto mask = torch::tensor(trace.actionMasks).reshape({rows, actionCount});
	auto acts = torch::tensor(trace.actions, torch::kLong);
	TORCH_CHECK((acts >= 0).all().item<bool>() && (acts < actionCount).all().item<bool>() &&
		(mask.gather(1, acts.unsqueeze(1)) != 0).all().item<bool>(), "Invalid recorded action");
	auto probs = torch::zeros({rows, actionCount}, torch::kFloat32);
	probs.scatter_(1, acts.unsqueeze(1), 1.f);
	std::vector<torch::Tensor> shard = {obs, mask, probs};
	torch::save(shard, path.string());
}

void GGL::InferUnit::SetRandomSeed(uint64_t seed) {
	torch::manual_seed(seed);
}

RLGC::Action GGL::InferUnit::InferAction(const RLGC::Player& player, const RLGC::GameState& state, bool deterministic, float temperature) {
	return BatchInferActions({ player }, { state }, deterministic, temperature)[0];
}

std::vector<RLGC::Action> GGL::InferUnit::BatchInferActions(const std::vector<RLGC::Player>& players, const std::vector<RLGC::GameState>& states, bool deterministic, float temperature, InferenceTrace* trace) {
	RG_ASSERT(players.size() > 0 && states.size() > 0);
	RG_ASSERT(players.size() == states.size());

	int batchSize = players.size();
	std::vector<float> allObs;
	std::vector<uint8_t> allActionMasks;
	for (int i = 0; i < batchSize; i++) {
		FList curObs = obsBuilder->BuildObs(players[i], states[i]);
		if (curObs.size() != obsSize) {
			RG_ERR_CLOSE(
				"InferUnit: Obs builder produced an obs that differs from the provided size (expected: " << obsSize << ", got: " << curObs.size() << ")\n" <<
				"Make sure you provided the correct obs size to the InferUnit constructor.\n" <<
				"Also, make sure there aren't an incorrect number of players (there are " << states[i].players.size() << " in this state)"
			);
		}
		allObs += curObs;

		allActionMasks += actionParser->GetActionMask(players[i], states[i]);
	}
	
	std::vector<RLGC::Action> results = {};

	try {
		RG_NO_GRAD;

		auto device = useGPU ? torch::kCUDA : torch::kCPU;

		auto tObs = torch::tensor(allObs).reshape({(int64_t)players.size(), obsSize});
		auto tActionMasks = torch::tensor(allActionMasks).reshape({(int64_t)players.size(), this->actionParser->GetActionAmount()});

		tObs = tObs.to(device);
		tActionMasks = tActionMasks.to(device);
		torch::Tensor tActions, tLogProbs;

		// Policy probabilities are clamped to a small positive value for PPO log/entropy
		// calculations. Never sample that numerical floor on a disabled action during
		// evaluation or winner-trace capture.
		auto probs = PPOLearner::InferPolicyProbsFromModels(*models, tObs, tActionMasks, temperature, false);
		auto allowed = tActionMasks.to(torch::kBool);
		TORCH_CHECK(allowed.any(-1).all().item<bool>(), "No legal actions for inference");
		probs = probs.masked_fill(allowed.logical_not(), 0);
		PPOLearner::ChooseActions(probs, deterministic, &tActions, &tLogProbs);

		auto actionIndices = TENSOR_TO_VEC<int>(tActions);
		if (trace) {
			trace->observations = allObs;
			trace->actionMasks = allActionMasks;
			trace->actions = actionIndices;
		}
		
		for (int i = 0; i < batchSize; i++) {
			RLGC::Action action = actionParser->ParseAction(actionIndices[i], players[i], states[i]);

			// The observation was of the mirrored game, so the action was picked for the mirrored game
			if (obsBuilder->IsMirroredX(players[i]))
				RLGC::ObsBuilder::MirrorActionX(action);

			results.push_back(action);
		}

	} catch (std::exception& e) {
		RG_ERR_CLOSE("InferUnit: Exception when inferring model: " << e.what());
	}

	return results;
}