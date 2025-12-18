#include "PredictMarkovChain.h"
#include <Search/MarkovChain.h>

namespace rai {

BanditProcess GT_BP_LGP(int planID) {
    // Get LGP transition data from DataManger
    auto& dataMgr = DataManger::getInstance();
    TransitionData data = dataMgr.getLGPTransitions(planID);
    
    // TODO: Convert TransitionData to Array<MarkovChain>
    MarkovChain mc(
        data.done_transitions,
        data.done_times,
        data.fail_transitions,
        data.fail_times,
        BanditType::LINE
    );
    Array<MarkovChain> chains;
    chains.append(mc);
    
    // TODO: Create and configure BanditProcess
    BanditProcess bp(chains);
    bp.nodeType = NodeType::LGPPathNode;
    
    return bp;
}

BanditProcess GT_BP_RRT(int planID, int numAction) {
    // Get RRT transition data from DataManger
    auto& dataMgr = DataManger::getInstance();
    auto [data, planLength] = dataMgr.getRRTTransitions(planID, numAction);
    
    // TODO: Convert TransitionData to Array<MarkovChain>
    Array<MarkovChain> chains;
    MarkovChain mc(
            data.done_transitions,
            data.done_times,
            data.fail_transitions,
            data.fail_times,
            BanditType::LINE
        );
    chains.append(mc);
    for(int i = numAction+1; i < planLength; ++i) {
        auto [data_i, planLength_i] = dataMgr.getRRTTransitions(planID, i);
        MarkovChain mc_i(
            data_i.done_transitions,
            data_i.done_times,
            data_i.fail_transitions,
            data_i.fail_times,
            BanditType::LINE
        );
        chains.append(mc_i);
    }
    data = dataMgr.getLGPTransitions(planID);
    MarkovChain mc_lgp(
        data.done_transitions,
        data.done_times,
        data.fail_transitions,
        data.fail_times,
        BanditType::LINE
    );
    chains.append(mc_lgp);

    
    // TODO: Create and configure BanditProcess
    BanditProcess bp(chains);
    bp.nodeType = NodeType::RRTNode;
    
    return bp;
}

BanditProcess GT_BP_Waypoints(int planID) {
    // Get waypoint transition data from DataManger
    auto& dataMgr = DataManger::getInstance();
    auto data = dataMgr.getWaypointTransitions(planID);
    MarkovChain mc(
        data.done_transitions,
        data.done_times,
        data.fail_transitions,
        data.fail_times,
        BanditType::LINE
    );
    Array<MarkovChain> chains;
    chains.append(mc);

    // Get planLength from first RRT call
    auto [data_0, planLength] = dataMgr.getRRTTransitions(planID, 0);
    for(int i = 0; i < planLength; ++i) {
        auto [data_i, planLength_i] = dataMgr.getRRTTransitions(planID, i);
        MarkovChain mc_i(
            data_i.done_transitions,
            data_i.done_times,
            data_i.fail_transitions,
            data_i.fail_times,
            BanditType::LINE
        );
        chains.append(mc_i);
    }
    data = dataMgr.getLGPTransitions(planID);
    MarkovChain mc_lgp(
        data.done_transitions,
        data.done_times,
        data.fail_transitions,
        data.fail_times,
        BanditType::LINE
    );
    chains.append(mc_lgp);

    
    // TODO: Create and configure BanditProcess
    BanditProcess bp(chains);
    bp.nodeType = NodeType::WaypointsNode;
    
    return bp;
}

BanditProcess myopic_GT_BP_Waypoints(int planID) {
    // Get waypoint transition data from DataManger
    auto& dataMgr = DataManger::getInstance();
    auto data = dataMgr.getWaypointTransitions(planID);
    
    // Only consider the waypoints chain itself (myopic)
    MarkovChain mc(
        data.done_transitions,
        data.done_times,
        data.fail_transitions,
        data.fail_times,
        BanditType::LINE
    );
    Array<MarkovChain> chains;
    chains.append(mc);
    
    // Create and configure BanditProcess
    BanditProcess bp(chains);
    bp.nodeType = NodeType::WaypointsNode;
    
    return bp;
}

BanditProcess myopic_GT_BP_RRT(int planID, int numAction) {
    // Get RRT transition data from DataManger
    auto& dataMgr = DataManger::getInstance();
    auto [data, planLength] = dataMgr.getRRTTransitions(planID, numAction);
    
    // Only consider the RRT chain itself (myopic)
    MarkovChain mc(
        data.done_transitions,
        data.done_times,
        data.fail_transitions,
        data.fail_times,
        BanditType::LINE
    );
    Array<MarkovChain> chains;
    chains.append(mc);
    
    // Create and configure BanditProcess
    BanditProcess bp(chains);
    bp.nodeType = NodeType::RRTNode;
    
    return bp;
}

} // namespace rai
