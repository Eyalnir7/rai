// #include "ModelPredictor.h"
#include "Search/GittinsNode.h"
// #include <Search/ProjectionBanditProcess.h>
#include <cmath> // for std::exp
// #include <Search/ModelPredictor.h>

using namespace rai;

// ModelPredictor& getModelPredictorSingleton() {
//     static ModelPredictor predictor("/home/eyal/projects/lgp-pddl/25-newSolvers/FolTest/Learning/trained_constraint_gnn_scripted.pt");
//     return predictor;
// }

// double predictFromModel(rai::Configuration& C, const StringAA& actionSequence) {
//     try {
//         ModelPredictor& predictor = getModelPredictorSingleton();
//         torch::Tensor prediction = predictor.predict(C, actionSequence);
//         return prediction.item<double>();
//     } catch (const std::exception& e) {
//         std::cerr << "Model prediction failed: " << e.what() << std::endl;
//         return 0.0;
//     }
// }

// // taskPlan to StringAA
// StringAA taskPlanToStringAA(const rai::TaskPlan& taskPlan) {
//     StringAA actionSequence;
//     for (const auto& action : taskPlan.actions) {
//         actionSequence.append(action.objects);
//     }
//     return actionSequence;
// }

void GittinsNode::initBanditProcess() {
    // Default implementation - can be overridden by derived classes
    // For now, create an empty BanditProcess
    banditProcess = std::make_unique<rai::BanditProcess>();
}

rai::TaskPlan GittinsNode::getTaskPlan() {
    return taskPlan;
}

// void GittinsNode::compute() {
//     if (rai::info().solver == "GITTINS") {
//         if(rai::info().verbose>0){
//             LOG(0) <<"compute at " <<name <<" ...";
//         }
//         c_now = -rai::cpuTime();
//         for(int i = 0; i <= stopping_time; i++) {
//             untimedCompute();
//         }
//         c_now += rai::cpuTime();
//         c += c_now;
//         backup_c(c_now);
//         if(l>1e9) isFeasible=false;
//         f_prio = computePriority();
//         if(rai::info().verbose>0){
//             if(isComplete) LOG(0) <<"computed " <<name <<" -> complete with c:" <<c <<" l:" <<l <<" level:" <<f_prio <<(isFeasible?" feasible":" INFEASIBLE") <<(isTerminal?" TERMINAL":0);
//             else LOG(0) <<"computed " <<name <<" -> still incomplete with c:" <<c;
//         }
//     } else {
//         // Use parent's implementation for non-GITTINS solvers
//         ComputeNode::compute();
//     }
// }

double GittinsNode::computePriority() {
    // if (rai::info().node_type == "ELS") {
    //     return baseLevel + computePenalty();
    // }
    if (rai::info().solver == "ELS") {
        return baseLevel + computePenalty();
    }

    if(!banditProcess){
        initBanditProcess();
    }

    int compute_units= c/banditProcess->sigma;
    // convert compute_units to double for more precise gittins index computation
    double compute_units_double = static_cast<double>(compute_units);
    auto [stopping_time, gittins_index] = banditProcess->compute_gittins_index(compute_units_double);
    // stopping_time = st; // Store stopping_time for use in compute()
    // cout << "stopping time: " << stopping_time << " gittins index: " << gittins_index << endl;

    return -gittins_index;
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