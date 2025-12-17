#pragma once

#include <Search/BanditProcess.h>

namespace rai {

// Ground Truth Bandit Process functions
BanditProcess GT_BP_LGP(int planID);
BanditProcess GT_BP_RRT(int planID, int numAction);
BanditProcess GT_BP_Waypoints(int planID);

} // namespace rai