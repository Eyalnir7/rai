#include "Search/GittinsNode.h"
#include "../LGP/LGP_computers.h"
#include <Search/ProjectionBanditProcess.h>

double GittinsNode::computePriority() {
    // if (rai::info().node_type == "ELS") {
    //     return baseLevel + computePenalty();
    // }
    if (rai::info().solver == "ELS") {
        return baseLevel + computePenalty();
    }
    rai::NodeType nodeType = getNodeType();
    taskPlan = getTaskPlan();

    if(nodeType==rai::NodeType::Other || nodeType==rai::NodeType::Skeleton || taskPlan.empty){
      if(nodeType==rai::NodeType::Skeleton) return -2;
      return 0;
    }

    if (!banditProcess) {
      banditProcess = std::make_unique<rai::ProjectionBanditProcess>(taskPlan, nodeType);
    }
    // std::cout << "using bandit process for Gittins index computation" << std::endl;
    return -banditProcess->compute_gittins_index(0);
}

// TODO: Store the A matrix (used in leonid calculation) of each node type in a file (Maybe use the graph structure of rai to have a single file of all matrices of every node type).
// In a separate file store the optimal stopping time for each node type.
// Have a singleton class that manages the A matrices and optimal stopping times. It gets the data only once and then provides it to the nodes by name when needed.
// Each node will have a name so one function can be used to get the A matrix and optimal stopping time for that node type.


// Essentially, the gittins search determines the optimal stopping time of each node and more importantly the task plan to be executed.
// After choosing a task plan, given that the distributions won't change, it will not change a task plan.
// Thus I want to see if the gittins index gives the same task plan as just choosing the task plan which minimizes the compute time.

// If the gittins search is too simple, first I can try to make changes to the distibution of the compute time during the search.
// Second, I can try implementing the compute tree from quim's phd. For gittins search on this, I would need to have some heuristic estimating the task plan from each node.