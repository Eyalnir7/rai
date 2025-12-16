#pragma once
#include "Search/BanditProcess.h"
#include <Search/PlanDataManager.h>
#include <Core/array.h>
#include <Search/NodeTypes.h>
#include <Search/TaskPlan.h>

namespace rai {

struct ProjectionBanditProcess : BanditProcess {
    const PlanDataManager::ComputeData* computeData;
    double nextLevelGI = 1;
    double currentGI = 0; // the gittins index from the first state
    int optimalStoppingTime = -1;
    rai::NodeType nodeType;
    rai::PlanDataManager& planDataManager = rai::PlanDataManager::getInstance();
    PlanDataManager::PlanData planData;

    ProjectionBanditProcess(const TaskPlan taskPlan, rai::NodeType nodeType);

    double compute_gittins_index(int state) const override;
    void updatePlanData(const TaskPlan taskPlan, rai::NodeType nodeType);
    void update_probs(const TaskPlan taskPlan, rai::NodeType nodeType);
};

} // namespace rai