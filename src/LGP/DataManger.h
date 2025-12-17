#pragma once

#include <string>
#include <map>
#include <vector>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

struct TransitionData {
    std::vector<double> done_transitions;
    std::vector<int> done_times;
    std::vector<double> fail_transitions;
    std::vector<int> fail_times;
};

class DataManger {
public:
    // Singleton access
    static DataManger& getInstance();
    
    // Delete copy constructor and assignment operator
    DataManger(const DataManger&) = delete;
    DataManger& operator=(const DataManger&) = delete;
    
    // Initialize with data directory path
    void initialize(const std::string& dataDir);
    
    // Get transition data for waypoints
    TransitionData getWaypointTransitions(int planID);
    
    // Get transition data for RRT
    std::pair<TransitionData, int> getRRTTransitions(int planID, int actionNum);
    
    // Get transition data for LGP
    TransitionData getLGPTransitions(int planID);
    
private:
    // Private constructor for singleton
    DataManger();
    
    // Load JSON data from files
    void loadData();
    
    // Helper to extract TransitionData from JSON
    TransitionData extractTransitionData(const json& j);
    
    // Data storage
    json waypointsData;
    json rrtData;
    json lgpData;
    
    // Data directory path
    std::string dataDirectory;
    bool initialized;
};
