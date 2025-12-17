#include <LGP/DataManger.h>
#include <fstream>
#include <iostream>
#include <sstream>

DataManger::DataManger() : initialized(false) {}

DataManger& DataManger::getInstance() {
    static DataManger instance;
    initialize(RAI_PARAM("", std::string, dataPath, "/Users/eyaltadmor/Documents/Thesis/lgp-pddl/25-newSolvers/FolTest/data/GTSim_copy/"));
    return instance;
}

void DataManger::initialize(const std::string& dataDir) {
    dataDirectory = dataDir;
    loadData();
    initialized = true;
}

void DataManger::loadData() {
    // Load waypoints_chains.json
    std::string waypointsPath = dataDirectory + "/waypoints_chains.json";
    std::ifstream waypointsFile(waypointsPath);
    if (!waypointsFile.is_open()) {
        std::cerr << "Error: Could not open " << waypointsPath << std::endl;
        return;
    }
    waypointsFile >> waypointsData;
    waypointsFile.close();
    
    // Load rrt_chains.json
    std::string rrtPath = dataDirectory + "/rrt_chains.json";
    std::ifstream rrtFile(rrtPath);
    if (!rrtFile.is_open()) {
        std::cerr << "Error: Could not open " << rrtPath << std::endl;
        return;
    }
    rrtFile >> rrtData;
    rrtFile.close();
    
    // Load lgp_chains.json
    std::string lgpPath = dataDirectory + "/lgp_chains.json";
    std::ifstream lgpFile(lgpPath);
    if (!lgpFile.is_open()) {
        std::cerr << "Error: Could not open " << lgpPath << std::endl;
        return;
    }
    lgpFile >> lgpData;
    lgpFile.close();
}

TransitionData DataManger::extractTransitionData(const json& j) {
    TransitionData data;
    
    if (j.contains("done_transitions")) {
        data.done_transitions = j["done_transitions"].get<std::vector<double>>();
    }
    
    if (j.contains("done_times")) {
        data.done_times = j["done_times"].get<std::vector<int>>();
    }
    
    if (j.contains("fail_transitions")) {
        data.fail_transitions = j["fail_transitions"].get<std::vector<double>>();
    }
    
    if (j.contains("fail_times")) {
        data.fail_times = j["fail_times"].get<std::vector<int>>();
    }
    
    return data;
}

TransitionData DataManger::getWaypointTransitions(int planID) {
    if (!initialized) {
        std::cerr << "Error: DataManger not initialized. Call initialize() first." << std::endl;
        return TransitionData();
    }
    
    std::string key = std::to_string(planID);
    if (waypointsData.contains(key)) {
        return extractTransitionData(waypointsData[key]);
    } else {
        std::cerr << "Warning: Plan ID " << planID << " not found in waypoints data" << std::endl;
        return TransitionData();
    }
}

std::pair<TransitionData, int> DataManger::getRRTTransitions(int planID, int actionNum) {
    if (!initialized) {
        std::cerr << "Error: DataManger not initialized. Call initialize() first." << std::endl;
        return TransitionData();
    }
    
    std::ostringstream keyStream;
    keyStream << planID << "_action_" << actionNum;
    std::string key = keyStream.str();
    
    if (rrtData.contains(key)) {
        if (rrtData[key].contains("planLength")){
            int planLength = rrtData[key]["planLength"].get<int>();
            return {extractTransitionData(rrtData[key]), planLength};
        }
    } else {
        std::cerr << "Warning: Key " << key << " not found in RRT data" << std::endl;
        return TransitionData();
    }
}

TransitionData DataManger::getLGPTransitions(int planID) {
    if (!initialized) {
        std::cerr << "Error: DataManger not initialized. Call initialize() first." << std::endl;
        return TransitionData();
    }
    
    std::string key = std::to_string(planID);
    if (lgpData.contains(key)) {
        return extractTransitionData(lgpData[key]);
    } else {
        std::cerr << "Warning: Plan ID " << planID << " not found in LGP data" << std::endl;
        return TransitionData();
    }
}
