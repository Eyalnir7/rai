#include <Search/PlanDataManager.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <limits>
#include <algorithm>
#include <utility>


namespace rai {

PlanDataManager::ComputeData PlanDataManager::ComputeData::slice(size_t i) const {
    ComputeData result;
    if(i > t.N) i = t.N; // bound check (assuming rai::Array has .N as size)

    result.t          = t({0, i});          // take [0, i)
    result.succ_probs = succ_probs({0, i});
    result.fail_probs = fail_probs({0, i});
    result.next_probs = next_probs({0, i});
    return result;
}

// Slice by condition on t: take all with t < threshold
PlanDataManager::ComputeData PlanDataManager::ComputeData::sliceByThreshold(double threshold) const {
    ComputeData result;
    size_t cutoff = 0;
    while(cutoff < t.N && t(cutoff) < threshold) {
        cutoff++;
    }
    return slice(cutoff);
}

    PlanDataManager& PlanDataManager::getInstance() {
        static PlanDataManager instance;
        return instance;
    }

    void PlanDataManager::loadPlansFile(const std::string& filepath) {
        try {
            std::cout << "Loading data from: " << filepath << std::endl;
            std::ifstream file(filepath);
            if (!file.is_open()) {
                std::cerr << "Error: Could not open file " << filepath << std::endl;
                initialized = false;
                throw std::runtime_error("Failed to open file");
            }
            
            Json::CharReaderBuilder builder;
            std::string errors;
            
            if (!Json::parseFromStream(builder, file, &jsonData, &errors)) {
                std::cerr << "Error parsing JSON: " << errors << std::endl;
                file.close();
                initialized = false;
                throw std::runtime_error("Failed to parse JSON");
            }
            
            file.close();
            
            // Extract single action plans
            extractSingleActionPlans();
            
            initialized = true;
            std::cout << "Successfully loaded plans from: " << filepath << std::endl;
            
        } catch (const std::exception& e) {
            std::cerr << "Error loading plans file " << filepath << ": " << e.what() << std::endl;
            initialized = false;
            throw;
        }
    }

    bool PlanDataManager::getComputeData(const std::string& planString, const NodeType nodeType, ComputeData& data) const {
        if (!initialized) {
            std::cerr << "PlanDataManager not initialized. Call loadPlansFile() first." << std::endl;
            return false;
        }
        
        // Find the plan node
        const Json::Value* planNode = findPlanNode(planString);
        if (!planNode) {
            return false;
        }
        
        // Find the node type within the plan
        const Json::Value* typeNode = findTypeNode(*planNode, toString(nodeType));
        if (!typeNode) {
            return false;
        }
        
        // Extract arrays from the type node
        bool success = extractComputeDataFromJson(*typeNode, data);
        
        return success;
    }

    bool PlanDataManager::getComputeData(const rai::TaskPlan& taskPlan, const NodeType nodeType, ComputeData& data) const {
        return getComputeData(taskPlan.toString(), nodeType, data);
    }

    bool PlanDataManager::getComputeData(const rai::Action& action, const NodeType nodeType, ComputeData& data) const {
        return getComputeData(action.toString(), nodeType, data);
    }

    bool PlanDataManager::hasData(const std::string& planString, const std::string& nodeType) const {
        if (!initialized) return false;
        
        const Json::Value* planNode = findPlanNode(planString);
        if (!planNode) return false;
        
        const Json::Value* typeNode = findTypeNode(*planNode, nodeType);
        return typeNode != nullptr;
    }

    rai::Array<rai::String> PlanDataManager::getNodeTypes(const std::string& planString) const {
        rai::Array<rai::String> types;
        
        if (!initialized) return types;
        
        const Json::Value* planNode = findPlanNode(planString);
        if (!planNode || !planNode->isObject()) return types;
        
        // Iterate through all node types in the plan
        auto memberNames = planNode->getMemberNames();
        for (const auto& typeName : memberNames) {
            types.append(rai::String(typeName.c_str()));
        }
        
        return types;
    }

    rai::Array<rai::TaskPlan> PlanDataManager::getPlans() const {
        rai::Array<rai::TaskPlan> allPlans;
        
        if (!initialized || !jsonData.isObject()) return allPlans;
        
        // Get all top-level plan keys
        auto memberNames = jsonData.getMemberNames();
        for (const std::string& planName : memberNames) {
            allPlans.append(TaskPlan(planName));
        }
        
        return allPlans;
    }

    rai::Array<rai::Action> PlanDataManager::getSingleActionPlans() const {
        rai::Array<rai::Action> singleActionPlansList;
        
        if (!initialized || !singleActionPlans.isObject()) return singleActionPlansList;
        
        // Get all single-action plan keys
        auto memberNames = singleActionPlans.getMemberNames();
        for (const std::string& planName : memberNames) {
            singleActionPlansList.append(Action(planName));
        }
        
        return singleActionPlansList;
    }

// Helper function to get plans sorted by distance to target plan
rai::Array<TaskPlan> PlanDataManager::getPlansSortedByDistance(const TaskPlan& targetPlan) const {
    if (!initialized) {
        std::cerr << "PlanDataManager not initialized. Call loadPlansFile() first." << std::endl;
        return rai::Array<TaskPlan>();
    }

    Array<TaskPlan> plans = getPlans();
    if (plans.N == 0) {
        std::cerr << "No plans available for sorting" << std::endl;
        return rai::Array<TaskPlan>();
    }
    
    // Create pairs of (distance, plan) for sorting
    rai::Array<std::pair<double, TaskPlan>> planDistances;
    
    for (uint i = 0; i < plans.N; i++) {
        double distance = planDistance(targetPlan, plans(i));
        planDistances.append(std::make_pair(distance, plans(i)));
    }
    
    // Sort by distance (ascending - closest first)
    std::sort(planDistances.p, planDistances.p + planDistances.N, 
              [](const std::pair<double, TaskPlan>& a, const std::pair<double, TaskPlan>& b) {
                  return a.first < b.first;
              });
    
    // Extract sorted plans
    rai::Array<TaskPlan> sortedPlans;
    for (uint i = 0; i < planDistances.N; i++) {
        sortedPlans.append(planDistances(i).second);
    }
    
    return sortedPlans;
}

// Helper function to get single actions sorted by distance to target action
rai::Array<Action> PlanDataManager::getActionsSortedByDistance(const Action& targetAction) const {
    if (!initialized) {
        std::cerr << "PlanDataManager not initialized. Call loadPlansFile() first." << std::endl;
        return rai::Array<Action>();
    }

    Array<Action> actions = getSingleActionPlans();
    if (actions.N == 0) {
        std::cerr << "No single actions available for sorting" << std::endl;
        return rai::Array<Action>();
    }
    
    // Create pairs of (distance, action) for sorting
    rai::Array<std::pair<double, Action>> actionDistances;
    
    for (uint i = 0; i < actions.N; i++) {
        double distance = actionDistance(targetAction, actions(i));
        actionDistances.append(std::make_pair(distance, actions(i)));
    }
    
    // Sort by distance (ascending - closest first)
    std::sort(actionDistances.p, actionDistances.p + actionDistances.N, 
              [](const std::pair<double, Action>& a, const std::pair<double, Action>& b) {
                  return a.first < b.first;
              });
    
    // Extract sorted actions
    rai::Array<Action> sortedActions;
    for (uint i = 0; i < actionDistances.N; i++) {
        sortedActions.append(actionDistances(i).second);
    }
    
    return sortedActions;
}

// Original function kept for backward compatibility
bool PlanDataManager::getPlanData(const TaskPlan& taskPlan, const NodeType nodeType, PlanData& data) const {
    return getPlanDataWithFallback(taskPlan, nodeType, data);
}

// Original function kept for backward compatibility  
void PlanDataManager::getProjectedPlanData(const TaskPlan& taskPlan, const NodeType nodeType, PlanData& data) const {
    getPlanDataWithFallback(taskPlan, nodeType, data);
}

// Improved function that searches through multiple plans for complete data
bool PlanDataManager::getPlanDataWithFallback(const TaskPlan& taskPlan, const NodeType nodeType, PlanData& data) const {
    if (!initialized) {
        std::cerr << "PlanDataManager not initialized. Call loadPlansFile() first." << std::endl;
        return false;
    }
    // cout << "Getting plan data with fallback for plan: " << taskPlan << " and node type: " << toString(nodeType) << endl;
    // Get all plans sorted by distance to our target plan
    Array<TaskPlan> sortedPlans = getPlansSortedByDistance(taskPlan);
    if (sortedPlans.N == 0) {
        std::cerr << "No plans available for fallback search" << std::endl;
        return false;
    }

    auto it = std::find(NODE_TYPE_ORDER.begin(), NODE_TYPE_ORDER.end(), nodeType);
    if (it == NODE_TYPE_ORDER.end()) {
        return false;
    }

    // For each required node type in order
    for(; it != NODE_TYPE_ORDER.end(); ++it) {
        NodeType currentType = *it;
        ComputeData computeData;
        bool foundData = false;

        if(currentType != NodeType::RRTNode) {
            // Search through sorted plans until we find data for this node type
            for (uint planIdx = 0; planIdx < sortedPlans.N; planIdx++) {
                const Json::Value* planNode = findPlanNode(sortedPlans(planIdx).toString());
                if (!planNode) continue;

                const Json::Value* typeNode = findTypeNode(*planNode, toString(currentType));
                if (!typeNode) continue;

                bool success = extractComputeDataFromJson(*typeNode, computeData);
                if (success) {
                    data.append(computeData);
                    foundData = true;
                    if (planIdx > 0) { // Only log if we had to use a fallback plan
                        // std::cout << "Found data for " << toString(currentType) 
                        //           << " in fallback plan: " << sortedPlans(planIdx) << std::endl;
                    }
                    break;
                }
            }

            if (!foundData) {
                std::cerr << "Failed to find compute data for type: " << toString(currentType) 
                          << " in any available plan" << std::endl;
                return false;
            }
        } else {
            // For RRTNode, we need to find data for each action
            for (const auto& action : taskPlan.actions) {
                bool foundActionData = false;
                
                if (getComputeData(action, currentType, computeData)) {
                    data.append(computeData);
                    foundActionData = true;
                }
                
                if (!foundActionData) {
                    // search in the fallback single action plans
                    Array<Action> sortedSingleAction = getActionsSortedByDistance(action);
                    for (const auto& fallbackAction : sortedSingleAction) {
                        if (getComputeData(fallbackAction, currentType, computeData)) {
                            data.append(computeData);
                            foundActionData = true;
                            break;
                        }
                    }
                }
                if(!foundActionData) {
                    std::cerr << "Failed to get compute data for action: " << action.toString() 
                              << " and type: " << toString(currentType) << " in any available plan" << std::endl;
                    return false;
                }
            }
        }
    }

    return true;
}

void PlanDataManager::clear() {
    jsonData.clear();
    singleActionPlans.clear();
    initialized = false;
}

// Helper methods
bool PlanDataManager::extractArrayFromJson(const Json::Value& node, const std::string& key, rai::Array<double>& array) const {
    if (!node.isMember(key)) {
        std::cerr << "Key '" << key << "' not found in JSON node" << std::endl;
        return false;
    }
    
    const Json::Value& jsonArray = node[key];
    if (!jsonArray.isArray()) {
        std::cerr << "Key '" << key << "' is not an array" << std::endl;
        return false;
    }
    
    array.clear();
    array.resize(jsonArray.size());
    
    for (Json::ArrayIndex i = 0; i < jsonArray.size(); ++i) {
        if (jsonArray[i].isNumeric()) {
            array(i) = jsonArray[i].asDouble();
        } else {
            std::cerr << "Array element " << i << " in '" << key << "' is not numeric" << std::endl;
            return false;
        }
    }
    
    return true;
}

bool PlanDataManager::extractComputeDataFromJson(const Json::Value& node, ComputeData& data) const {
    bool success = true;
    success &= extractArrayFromJson(node, "t", data.t);
    success &= extractArrayFromJson(node, "succ_probs", data.succ_probs);
    success &= extractArrayFromJson(node, "fail_probs", data.fail_probs);
    success &= extractArrayFromJson(node, "next_probs", data.next_probs);
    return success;
}

const Json::Value* PlanDataManager::findPlanNode(const std::string& planString) const {
    if (!jsonData.isObject() || !jsonData.isMember(planString)) {
        return nullptr;
    }
    return &jsonData[planString];
}

const Json::Value* PlanDataManager::findTypeNode(const Json::Value& planNode, const std::string& nodeType) const {
    if (!planNode.isObject() || !planNode.isMember(nodeType)) {
        return nullptr;
    }
    return &planNode[nodeType];
}

void PlanDataManager::extractSingleActionPlans() {
    singleActionPlans.clear();
    
    if (!jsonData.isObject()) {
        return;
    }
    
    // Iterate through all plans in jsonData
    auto memberNames = jsonData.getMemberNames();
    for (const std::string& planName : memberNames) {
        // Count the number of actions in the plan by counting parentheses pairs
        // Single action plans have the format "(action_name arg1 arg2 ...)"
        size_t actionCount = 0;
        size_t pos = 0;
        
        while ((pos = planName.find('(', pos)) != std::string::npos) {
            actionCount++;
            pos++;
            if(actionCount == 2) break;
        }
        
        // If this plan contains exactly one action, add it to singleActionPlans
        if (actionCount == 1) {
            singleActionPlans[planName] = jsonData[planName];
        }
    }
    
    std::cout << "Extracted " << singleActionPlans.size() << " single-action plans" << std::endl;
}

} // namespace rai