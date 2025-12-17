#include "PredictMarkovChain.h"
#include <Search/DataManger.h>
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
    chains.push_back(mc);
    
    // TODO: Create and configure BanditProcess
    BanditProcess bp(chains);
    bp.nodeType = NodeType::LGP;
    
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
    chains.push_back(mc);
    for(int i = numAction+1; i < planLength; ++i) {
        auto [data, planLength] = dataMgr.getRRTTransitions(planID, i);
        MarkovChain mc(
            data.done_transitions,
            data.done_times,
            data.fail_transitions,
            data.fail_times,
            BanditType::LINE
        );
        chains.push_back(mc);
    }
    data = dataMgr.getLGPTransitions(planID);
    MarkovChain mc_lgp(
        data.done_transitions,
        data.done_times,
        data.fail_transitions,
        data.fail_times,
        BanditType::LINE
    );
    chains.push_back(mc_lgp);

    
    // TODO: Create and configure BanditProcess
    BanditProcess bp(chains);
    bp.nodeType = NodeType::RRT;
    
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
    chains.push_back(mc);

    for(int i = 0; i < planLength; ++i) {
        auto [data, planLength] = dataMgr.getRRTTransitions(planID, i);
        MarkovChain mc(
            data.done_transitions,
            data.done_times,
            data.fail_transitions,
            data.fail_times,
            BanditType::LINE
        );
        chains.push_back(mc);
    }
    data = dataMgr.getLGPTransitions(planID);
    MarkovChain mc_lgp(
        data.done_transitions,
        data.done_times,
        data.fail_transitions,
        data.fail_times,
        BanditType::LINE
    );
    chains.push_back(mc_lgp);

    
    // TODO: Create and configure BanditProcess
    BanditProcess bp(chains);
    bp.nodeType = NodeType::WaypointsNode;
    
    return bp;
}

} // namespace rai
