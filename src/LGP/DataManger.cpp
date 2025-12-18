#include <LGP/DataManger.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <Core/util.h>

DataManger::DataManger() : initialized(false) {
    rai::String dataPath = rai::getParameter<rai::String>("dataPath", "/home/eyal/Documents/code_repos/lgp-pddl/25-newSolvers/FolTest/data/GTSimICT");
    initialize(dataPath.p);
}

DataManger& DataManger::getInstance() {
    static DataManger instance;
    return instance;
}

void DataManger::initialize(const std::string& dataDir) {
    dataDirectory = dataDir;
    loadData();
    initialized = true;
}

void DataManger::loadData() {
    Json::Reader reader;
    
    // Load waypoints_chains.json
    std::string waypointsPath = dataDirectory + "/waypoints_chains.json";
    std::ifstream waypointsFile(waypointsPath);
    if (!waypointsFile.is_open()) {
        std::cerr << "Error: Could not open " << waypointsPath << std::endl;
        return;
    }
    if (!reader.parse(waypointsFile, waypointsData)) {
        std::cerr << "Error: Failed to parse " << waypointsPath << std::endl;
    }
    waypointsFile.close();
    
    // Load rrt_chains.json
    std::string rrtPath = dataDirectory + "/rrt_chains.json";
    std::ifstream rrtFile(rrtPath);
    if (!rrtFile.is_open()) {
        std::cerr << "Error: Could not open " << rrtPath << std::endl;
        return;
    }
    if (!reader.parse(rrtFile, rrtData)) {
        std::cerr << "Error: Failed to parse " << rrtPath << std::endl;
    }
    rrtFile.close();
    
    // Load lgp_chains.json
    std::string lgpPath = dataDirectory + "/lgp_chains.json";
    std::ifstream lgpFile(lgpPath);
    if (!lgpFile.is_open()) {
        std::cerr << "Error: Could not open " << lgpPath << std::endl;
        return;
    }
    if (!reader.parse(lgpFile, lgpData)) {
        std::cerr << "Error: Failed to parse " << lgpPath << std::endl;
    }
    lgpFile.close();
}

TransitionData DataManger::extractTransitionData(const Json::Value& j) {
    TransitionData data;
    
    if (j.isMember("done_transitions") && j["done_transitions"].isArray()) {
        for (const auto& val : j["done_transitions"]) {
            data.done_transitions.push_back(val.asDouble());
        }
    }
    
    if (j.isMember("done_times") && j["done_times"].isArray()) {
        for (const auto& val : j["done_times"]) {
            data.done_times.push_back(val.asInt());
        }
    }
    
    if (j.isMember("fail_transitions") && j["fail_transitions"].isArray()) {
        for (const auto& val : j["fail_transitions"]) {
            data.fail_transitions.push_back(val.asDouble());
        }
    }
    
    if (j.isMember("fail_times") && j["fail_times"].isArray()) {
        for (const auto& val : j["fail_times"]) {
            data.fail_times.push_back(val.asInt());
        }
    }
    
    return data;
}

TransitionData DataManger::getWaypointTransitions(int planID) {
    if (!initialized) {
        std::cerr << "Error: DataManger not initialized. Call initialize() first." << std::endl;
        return TransitionData();
    }
    
    std::string key = std::to_string(planID);
    if (waypointsData.isMember(key)) {
        return extractTransitionData(waypointsData[key]);
    } else {
        std::cerr << "Warning: Plan ID " << planID << " not found in waypoints data" << std::endl;
        return TransitionData();
    }
}

std::pair<TransitionData, int> DataManger::getRRTTransitions(int planID, int actionNum) {
    if (!initialized) {
        std::cerr << "Error: DataManger not initialized. Call initialize() first." << std::endl;
        return {TransitionData(),-1};
    }
    
    std::ostringstream keyStream;
    keyStream << planID << "_action_" << actionNum;
    std::string key = keyStream.str();
    
    if (rrtData.isMember(key)) {
        if (rrtData[key].isMember("planLength")){
            int planLength = rrtData[key]["planLength"].asInt();
            return {extractTransitionData(rrtData[key]), planLength};
        }
    } else {
        std::cerr << "Warning: Key " << key << " not found in RRT data" << std::endl;
        return {TransitionData(),-1};
    }
}

TransitionData DataManger::getLGPTransitions(int planID) {
    if (!initialized) {
        std::cerr << "Error: DataManger not initialized. Call initialize() first." << std::endl;
        return TransitionData();
    }
    
    std::string key = std::to_string(planID);
    if (lgpData.isMember(key)) {
        return extractTransitionData(lgpData[key]);
    } else {
        std::cerr << "Warning: Plan ID " << planID << " not found in LGP data" << std::endl;
        return TransitionData();
    }
}
