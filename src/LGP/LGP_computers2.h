/*  ------------------------------------------------------------------
    Copyright (c) 2011-2024 Marc Toussaint
    email: toussaint@tu-berlin.de

    This code is distributed under the MIT License.
    Please see <root-path>/LICENSE for details.
    --------------------------------------------------------------  */

#pragma once

#include "../Search/ComputeNode.h"
#include "../Search/AStar.h"
#include "../Logic/folWorld.h"
#include "../Optim/NLP_GraphSolver.h"
#include "../PathAlgos/RRT_PathFinder.h"
#include "LGP_TAMP_Abstraction.h"
#include "PredictMarkovChain.h"
#include <Search/GittinsNode.h>
#include <Search/NodeTypes.h>
#include <regex>

namespace rai {

//===========================================================================

struct LGP2_GlobalInfo {
  RAI_PARAM("LGP/", int, verbose, 0)
  RAI_PARAM("LGP/", double, skeleton_w0, 1.)
  RAI_PARAM("LGP/", double, skeleton_wP, 2.)
  RAI_PARAM("LGP/", double, waypoint_w0, 10.)
  RAI_PARAM("LGP/", double, waypoint_wP, 2.)
  RAI_PARAM("LGP/", int, waypointStopEvals, 1000)
  RAI_PARAM("LGP/", int, rrtStopEvals, 10000)
  RAI_PARAM("LGP/", double, rrtStepsize, .1)
  RAI_PARAM("LGP/", double, rrtTolerance, .03)
  RAI_PARAM("LGP/", double, pathCtrlCosts, 1.)
  RAI_PARAM("LGP/", int, pathStepsPerPhase, 30)
  RAI_PARAM("LGP/", double, collScale, 1e1)
  RAI_PARAM("LGP/", bool, useSequentialWaypointSolver, false)
  RAI_PARAM("", rai::String, solver, "ELS")
  RAI_PARAM("LGP/", double, quantum, 0.01)
  RAI_PARAM("", int, numTaskPlans, 20)
  RAI_PARAM("", rai::String, predictionType, "myopicGT") // GT / myopicGT / none / GNN
  RAI_PARAM("GNN/", rai::String, modelsDir, "models/")
  RAI_PARAM("GITTINS", int, numWaypoints, 100)
};

//===========================================================================

struct LGPComp2_root : GittinsNode {
  // FOL_World& L;
  Configuration& C;
  LGP_TAMP_Abstraction& tamp;
  int runSeed;
  // bool useBroadCollisions=false;
  // StringA explicitCollisions;
  // StringA explicitLift;
  // String explicitTerminalSkeleton;
  // std::shared_ptr<rai::AStar> fol_astar;
  std::shared_ptr<LGP2_GlobalInfo> info;
  bool fixedSkeleton=false;
  int numSkeletonsTried=0;
  
  // NodePredictor manages all prediction logic (GT, myopicGT, GNN)
  std::shared_ptr<NodePredictor> predictor;

  LGPComp2_root(Configuration& _C, LGP_TAMP_Abstraction& _tamp, const StringA& explicitLift, const String& explicitTerminalSkeleton, int _runSeed, std::shared_ptr<NodePredictor> _predictor = nullptr);

  virtual rai::Configuration* getConfiguration() override { return &C; }
  virtual void untimedCompute() {}
  // virtual int getNumDecisions(){ return (info->solver == "GITTINS") ? info->numTaskPlans : -1; };
    virtual int getNumDecisions(){ return -1; };
//    virtual double effortHeuristic(){ return 11.+10.; }
  virtual double branchingPenalty_child(int i);

  virtual std::shared_ptr<ComputeNode> createNewChild(int i);

  int verbose() { return info->verbose; }
};

//===========================================================================

struct LGPComp2_Skeleton : GittinsNode {
  LGPComp2_root* root=0;
  int num=0;
  // Skeleton skeleton;
  //shared_ptr<SkeletonSolver> sol;
  // StringA actionSequence;
  Array<StringA> actionSequence;
  bool triedChild = false;
  // Array<Graph*> states;
  // arr times;

  LGPComp2_Skeleton(LGPComp2_root* _root, int num);
  // LGPComp2_Skeleton(LGPComp2_root* _root, const Skeleton& _skeleton);

  void createNLPs(const rai::Configuration& C);

  virtual rai::Configuration* getConfiguration() override { return &root->C; }
  virtual void untimedCompute();
  virtual rai::NodeType getNodeType() override { return rai::NodeType::Skeleton; }
  virtual rai::TaskPlan getTaskPlan() override { return rai::TaskPlan(actionSequence); }
  virtual void initBanditProcess() override;

  virtual int getNumDecisions() { return (root->info->solver == "GITTINS") ? root->info->numWaypoints : -1; }
//    virtual double branchingHeuristic(){ return root->info->waypoint_w0; }
//    virtual double effortHeuristic(){ return 10.+10.; }
  virtual double branchingPenalty_child(int i);
  double computePriority() override;

  virtual std::shared_ptr<ComputeNode> createNewChild(int i);
};

//===========================================================================

struct LGPComp2_FactorBounds: GittinsNode {
  LGPComp2_Skeleton* sket;
  int seed=0;
  KOMO komoWaypoints;
  std::shared_ptr<NLP_Factored> nlp;
  uint t=0;

  LGPComp2_FactorBounds(LGPComp2_Skeleton* _sket, int rndSeed);

  virtual void untimedCompute();
//    virtual double effortHeuristic(){ return 10.+1.*(komoWaypoints.T); }
  virtual int getNumDecisions() { return 1; }
  virtual std::shared_ptr<ComputeNode> createNewChild(int i);
};

//===========================================================================

struct LGPComp2_PoseBounds: GittinsNode {
  LGPComp2_Skeleton* sket;
  int seed=0;
  uint t=0;

  LGPComp2_PoseBounds(LGPComp2_Skeleton* _sket, int rndSeed);

  virtual void untimedCompute();
//    virtual double effortHeuristic(){ return 10.+1.*(sket->states.N); }
  virtual int getNumDecisions() { return 1; }
  virtual std::shared_ptr<ComputeNode> createNewChild(int i);
};

//===========================================================================

struct LGPComp2_Waypoints : GittinsNode {
  LGPComp2_Skeleton* sket;
  int seed=0;
  std::shared_ptr<KOMO> komoWaypoints;
  NLP_Solver sol;
  NLP_GraphSolver gsol;
  std::shared_ptr<LGP2_GlobalInfo> info = make_shared<LGP2_GlobalInfo>();

  LGPComp2_Waypoints(LGPComp2_Skeleton* _sket, int rndSeed);

  virtual rai::Configuration* getConfiguration() override { return &sket->root->C; }
  virtual void initBanditProcess() override;
  virtual void untimedCompute();
  virtual rai::NodeType getNodeType() override { return rai::NodeType::WaypointsNode; }
  virtual rai::TaskPlan getTaskPlan() override { return rai::TaskPlan(sket->actionSequence); }
//    virtual double effortHeuristic(){ return 10.+1.*(komoWaypoints->T); }
  virtual int getNumDecisions();
  virtual double branchingPenalty_child(int i);
  virtual std::shared_ptr<ComputeNode> createNewChild(int i);
};

//===========================================================================

struct LGPComp2_RRTpath : GittinsNode {
  LGPComp2_Waypoints* ways=0;
  LGPComp2_RRTpath* prev=0;
  LGPComp2_Skeleton *sket=0;
  int seed = 0;
  std::shared_ptr<LGP2_GlobalInfo> info = make_shared<LGP2_GlobalInfo>();

  shared_ptr<Configuration> C;
  uint t;
  shared_ptr<RRT_PathFinder> rrt;
  arr q0, qT;
  arr path;

  LGPComp2_RRTpath(ComputeNode* _par, LGPComp2_Waypoints* _ways, uint _t, int rndSeed);

  virtual void initBanditProcess() override;
  virtual void untimedCompute();
  virtual rai::NodeType getNodeType() override { return rai::NodeType::RRTNode; }
  virtual rai::TaskPlan getTaskPlan() override { return rai::TaskPlan(sket->actionSequence); }
  virtual double branchingPenalty_child(int i);

//    virtual double effortHeuristic(){ return 10.+1.*(ways->komoWaypoints->T-t-1); }

  virtual int getNumDecisions(){ return 1; }
  virtual std::shared_ptr<ComputeNode> createNewChild(int i);
};

//===========================================================================

struct LGPComp2_OptimizePath : GittinsNode {
  LGPComp2_Skeleton* sket=0;
  LGPComp2_Waypoints* ways=0;
  int seed=0;
  std::shared_ptr<LGP2_GlobalInfo> info = make_shared<LGP2_GlobalInfo>();

  shared_ptr<KOMO> komoPath;
  NLP_Solver sol;

  LGPComp2_OptimizePath(LGPComp2_Skeleton* _sket); //compute path from skeleton directly, without waypoints first
  LGPComp2_OptimizePath(LGPComp2_Waypoints* _ways); //compute path initialized from waypoints
  LGPComp2_OptimizePath(LGPComp2_RRTpath* _par, LGPComp2_Waypoints* _ways, int rndSeed); //compute path initialized from series of RRT solutions

  virtual void initBanditProcess() override;
  virtual void untimedCompute();
  virtual rai::NodeType getNodeType() override { return rai::NodeType::LGPPathNode; }
  virtual rai::TaskPlan getTaskPlan() override { return rai::TaskPlan(sket->actionSequence); }
//    virtual double effortHeuristic(){ return 0.; }

  virtual double sample() {
    CHECK(sol.ret, "");
    CHECK(sol.ret->done, "");
    if(sol.ret->ineq>1. || sol.ret->eq>4.) return 1e10;
    return sol.ret->ineq+sol.ret->eq;
  }

//        if(opt.verbose>0) LOG(0) <<*ret;
//        if(opt.verbose>1) cout <<komoPath.report(false, true, opt.verbose>2);
//        if(opt.verbose>0) komoPath.view(opt.verbose>1, STRING("optimized path\n" <<*ret));
//        //komoPath.checkGradients();
//        if(opt.verbose>1) komoPath.view_play(opt.verbose>2);
  virtual int getNumDecisions() { return 0; }
  virtual std::shared_ptr<ComputeNode> createNewChild(int i) { HALT("is terminal"); }
};

//===========================================================================

}//namespace
