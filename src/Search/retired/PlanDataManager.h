#pragma once
#include <Core/array.h>
#include <Core/util.h>
#include <string>
#include <memory>
#include <Search/TaskPlan.h>
#include <Search/NodeTypes.h>
#include <jsoncpp/json/json.h>

namespace rai {

class PlanDataManager {
public:
    struct ComputeData {
        rai::Array<double> t;
        rai::Array<double> succ_probs;
        rai::Array<double> fail_probs;
        rai::Array<double> next_probs;

        ComputeData slice(size_t i) const;
        ComputeData sliceByThreshold(double threshold) const;
    };
    using PlanData = rai::Array<ComputeData>;
    // Singleton access
    static PlanDataManager& getInstance();
    
    // Initialize with plans.json file
    void loadPlansFile(const std::string& filepath);
    
    // Get plan data for a specific plan and node type
    bool getComputeData(const std::string& planString, const NodeType nodeType, ComputeData& data) const;
    bool getComputeData(const TaskPlan& taskPlan, const NodeType nodeType, ComputeData& data) const;
    bool getComputeData(const rai::Action& action, const NodeType nodeType, ComputeData& data) const;
    
    // Original plan data functions (deprecated - use fallback versions)
    bool getPlanData(const TaskPlan& taskPlan, const NodeType nodeType, PlanData& data) const;
    void getProjectedPlanData(const TaskPlan& taskPlan, const NodeType nodeType, PlanData& data) const;
    
    // Improved plan data functions with fallback strategy
    bool getPlanDataWithFallback(const TaskPlan& taskPlan, const NodeType nodeType, PlanData& data) const;
    
    // Helper function to get plans sorted by distance
    rai::Array<TaskPlan> getPlansSortedByDistance(const TaskPlan& targetPlan) const;
    
    // Helper function to get single actions sorted by distance to target action
    rai::Array<Action> getActionsSortedByDistance(const Action& targetAction) const;

    // Check if data exists for a plan and node type
    bool hasData(const std::string& planString, const std::string& nodeType) const;
    
    // Get all available node types for a plan
    rai::Array<rai::String> getNodeTypes(const std::string& planString) const;
    
    // Get all available plans
    rai::Array<rai::TaskPlan> getPlans() const;
    
    // Get all single-action plans
    rai::Array<rai::Action> getSingleActionPlans() const;
    
    // Clear all loaded data
    void clear();
    
    // Check if manager is initialized
    bool isInitialized() const { return initialized; }

private:
    // Private constructor for singleton
    PlanDataManager() = default;
    ~PlanDataManager() = default;
    
    // Delete copy constructor and assignment operator
    PlanDataManager(const PlanDataManager&) = delete;
    PlanDataManager& operator=(const PlanDataManager&) = delete;
    
    // Internal data storage
    Json::Value jsonData;
    Json::Value singleActionPlans;
    bool initialized = false;
    
    // Helper methods
    bool extractArrayFromJson(const Json::Value& node, const std::string& key, rai::Array<double>& array) const;
    bool extractComputeDataFromJson(const Json::Value& node, ComputeData& data) const;
    const Json::Value* findPlanNode(const std::string& planString) const;
    const Json::Value* findTypeNode(const Json::Value& planNode, const std::string& nodeType) const;
    void extractSingleActionPlans();
};

} // namespace rai