/*  ------------------------------------------------------------------
    Copyright (c) 2011-2024 Marc Toussaint
    email: toussaint@tu-berlin.de

    This code is distributed under the MIT License.
    Please see <root-path>/LICENSE for details.
    --------------------------------------------------------------  */

#include "LGP_computers2.h"

#include "../Kin/viewer.h"
#include "../Gui/opengl.h"
#include <iomanip>
#include "../Kin/F_qFeatures.h"
#include "../Optim/constrained.h"
#include "../KOMO/pathTools.h"

rai::LGPComp2_root::
    LGPComp2_root(Configuration& _C, LGP_TAMP_Abstraction& _tamp, const StringA& explicitLift, const String& explicitTerminalSkeleton)
    : GittinsNode(0), C(_C), tamp(_tamp) {
  name <<"LGPComp2_root#0";
  info = make_shared<LGP2_GlobalInfo>();
  isComplete = true;
}

double rai::LGPComp2_root::branchingPenalty_child(int i) {
  return ::pow(double(i)/info->skeleton_w0, info->skeleton_wP);
}

std::shared_ptr<rai::ComputeNode> rai::LGPComp2_root::createNewChild(int i) {
  // cout << "Creating new LGPComp2_Skeleton child from root, child index: " << i << endl;
  return make_shared<LGPComp2_Skeleton>(this, i);
}

//===========================================================================

rai::LGPComp2_Skeleton::LGPComp2_Skeleton(rai::LGPComp2_root* _root, int num) : GittinsNode(_root), root(_root), num(num) {
  name <<"LGPComp2_Skeleton#"<<num;
  // if(_root->info->solver == "GITTINS"){
  //   needsWidening = false;
  // }
  // else{
  //   needsWidening = true;
  // }
}

void rai::LGPComp2_Skeleton::createNLPs(const Configuration& C) {
  NIY;
}

void rai::LGPComp2_Skeleton::untimedCompute() {
  //-- get next astar solution
  actionSequence = root->tamp.getNewActionSequence();
  if(!actionSequence.N){
    LOG(-1) <<"astar found no new skeleton";
    isComplete=true;
    isFeasible=false;
    return;
  }

  isComplete=true;
  l=0.;

  if(root->info->verbose>0) LOG(0) <<"FOL action sequence:" <<actionSequence;
  // if(root->info->verbose>1) LOG(0) <<skeleton;
}

double rai::LGPComp2_Skeleton::branchingPenalty_child(int i) {
  return ::pow(double(i)/root->info->waypoint_w0, root->info->waypoint_wP);
}

std::shared_ptr<rai::ComputeNode> rai::LGPComp2_Skeleton::createNewChild(int i) {
  return make_shared<LGPComp2_Waypoints>(this, i);
}

//===========================================================================

rai::LGPComp2_PoseBounds::LGPComp2_PoseBounds(rai::LGPComp2_Skeleton* _sket, int rndSeed)
  : GittinsNode(_sket), sket(_sket), seed(rndSeed) {
  name <<"LGPComp2_PoseBounds#"<<seed;
}

void rai::LGPComp2_PoseBounds::untimedCompute() {
  t++;
  NIY;
  std::shared_ptr<KOMO> komo = make_shared<KOMO>();

  komo->initRandom(0);

  NLP_Solver sol;
  sol.setProblem(komo->nlp());
  sol.setInitialization(komo->x);
  sol.solveStepping();

  if(!sol.ret->feasible) {
    l = 1e10;
    isComplete = true;
    return;
  }

  if(true){ NIY; //if(t==sket->states.N-1) {
    l = 0.;
    isComplete = true;
  }
}

std::shared_ptr<rai::ComputeNode> rai::LGPComp2_PoseBounds::createNewChild(int i) {
  return make_shared<LGPComp2_Waypoints>(sket, i);
}

//===========================================================================

rai::LGPComp2_FactorBounds::LGPComp2_FactorBounds(rai::LGPComp2_Skeleton* _sket, int rndSeed)
  : GittinsNode(_sket), sket(_sket), seed(rndSeed) {
  name <<"LGPComp2_FactorBounds#"<<seed;

  NIY; //komoWaypoints.clone(*sket->skeleton.komoWaypoints);
  komoWaypoints.initRandom(0);

  nlp = komoWaypoints.nlp_FactoredTime();
  CHECK_EQ(nlp->variableDimensions.N, komoWaypoints.T, "");
}

void rai::LGPComp2_FactorBounds::untimedCompute() {
  LGPComp2_root* root=sket->root;

  nlp->subSelect({t}, {});

  CHECK_EQ(nlp->dimension, komoWaypoints.x.N, "");

  //if(root->verbose()>1) komo->view(root->verbose()>2, STRING(name <<" - init"));

  NLP_Solver sol;
  sol.setProblem(nlp);
  sol.setInitialization(komoWaypoints.x);
  sol.solveStepping();

  if(!sol.ret->feasible) {
    l = 1e10;
    isComplete = true;
    if(root->verbose()>1) komoWaypoints.view_close();
    return;
  }

  t++;
  if(t==komoWaypoints.T) {
    isComplete = true;
    if(root->verbose()>1) komoWaypoints.view_close();
  }
}

std::shared_ptr<rai::ComputeNode> rai::LGPComp2_FactorBounds::createNewChild(int i) {
  return make_shared<LGPComp2_Waypoints>(sket, i);
}

//===========================================================================

rai::LGPComp2_Waypoints::LGPComp2_Waypoints(rai::LGPComp2_Skeleton* _sket, int rndSeed)
  : GittinsNode(_sket), sket(_sket), seed(rndSeed) {
  name <<"LGPComp2_Waypoints#"<<seed;
  taskPlan = TaskPlan(sket->actionSequence);

  LGPComp2_root* root=sket->root;

  komoWaypoints = root->tamp.get_waypointsProblem(root->C, sket->actionSequence);

  rnd.seed(rndSeed);
  // komoWaypoints->initRandom(0);
  komoWaypoints->run_prepare(komoWaypoints->opt.waypointsInitNoise);
  if(root->verbose()>2){
    komoWaypoints->view(root->verbose()>3, STRING(name <<" - init"));
  }

  sol.setProblem(komoWaypoints->nlp());
  sol.setOptions(OptOptions().set_stopEvals(sket->root->info->waypointStopEvals));
  sol.setInitialization(komoWaypoints->x);

}

rai::TaskPlan rai::LGPComp2_Waypoints::getTaskPlan() {
  if(!taskPlan.empty)  return taskPlan;
  taskPlan = TaskPlan(sket->actionSequence);
  return taskPlan;
}

void rai::LGPComp2_Waypoints::untimedCompute() {
  LGPComp2_root* root=sket->root;

  std::shared_ptr<SolverReturn> ret;
  if(sket->root->info->useSequentialWaypointSolver) {
    CHECK(!komoWaypoints->computeCollisions, "useSequentialWaypointSolver doesn't work with genericCollisions")
    gsol.solveInOrder();
    ret = gsol.ret;
  } else {
    double quantum = sket->root->info->quantum;
    auto startTime = rai::cpuTime();
    while(rai::cpuTime() - startTime < quantum) {
      if(sol.step()) break;
    }
    ret = sol.ret;
  }

  l = sol.ret->eq + sol.ret->ineq;
  isComplete = ret->done;

  //    checkJacobianCP(*komoWaypoints->nlp_SparseNonFactored(), komoWaypoints->x, 1e-6);
  if(root->verbose()>0) LOG(0) <<"ways " <<*ret;
  //if(root->verbose()>1) cout <<komoWaypoints->report(false, true, root->verbose()>3);
  if(root->verbose()>3) {
    // komoWaypoints->pathConfig.coll_reportProxies();
    komoWaypoints->pathConfig.reportLimits();
    // komoWaypoints->checkGradients();
    cout <<sol.optCon->L.reportGradients(komoWaypoints->featureNames) <<endl;
  }

  if(!isComplete && root->verbose()>4) {
    komoWaypoints->view(root->verbose()>5, STRING(name <<" - intermediate results, c:" <<c <<"\n" <<*ret));
    if(root->verbose()>5) komoWaypoints->view_play(true);
  }

  if(isComplete) {
    if(!((ret->ineq<.1) && (ret->eq<.1))) {
      isFeasible = false;
      komoWaypoints->view_close();
      if(root->verbose()>1) {
        cout <<sol.optCon->L.reportGradients(komoWaypoints->featureNames) <<endl;
      }
      // cout << "waypoints NOT feasible: ineq=" <<ret->ineq <<' ' <<"eq=" <<ret->eq <<endl;
    } else {
      // cout << "waypoints feasible!" <<endl;
      isFeasible = true;
      if(root->verbose()>2) komoWaypoints->view(root->verbose()>3, STRING(name <<" - final, c:" <<c <<"\n" <<*ret));
      if(root->verbose()>3) komoWaypoints->view_play(true);
      //    if(root->verbose()>1){
      //      static int count=0;
      //      if(ret->feasible)  write_png(komoWaypoints->pathConfig.gl()->getScreenshot(), STRING("z.vid/"<<std::setw(4)<<std::setfill('0')<<count++<<".png"));
      //    }
    }
  }
}

double rai::LGPComp2_Waypoints::branchingPenalty_child(int i) {
  return 0.0;
}

int rai::LGPComp2_Waypoints::getNumDecisions() { return 1; }

std::shared_ptr<rai::ComputeNode> rai::LGPComp2_Waypoints::createNewChild(int i) {
  //komoWaypoints->checkConsistency();
//  if(i==0){
//    return make_shared<LGPComp2_OptimizePath>(this);
//  }else if(i==1){
  return make_shared<LGPComp2_RRTpath>(this, this, 0, i);
//  }else HALT("only 2 options for child below waypoints");
  return std::shared_ptr<rai::ComputeNode>();
}

//===========================================================================

rai::LGPComp2_RRTpath::LGPComp2_RRTpath(ComputeNode* _par, rai::LGPComp2_Waypoints* _ways, uint _t, int rndSeed)
  : GittinsNode(_par), ways(_ways), sket(_ways->sket), t(_t) {
  if(!t) CHECK_EQ(_par, _ways, "");
  name <<"LGPComp2_RRTpath#" <<ways->seed <<'.' <<t;
  LGPComp2_root* root = ways->sket->root;
  rnd.seed(rndSeed);
  isTerminal = (t+1 >= ways->komoWaypoints->T);
  // cout << "isTerminal: " << isTerminal << endl;

  if(root->verbose()>1) LOG(0) <<"rrt for phase:" <<t;
  C = make_shared<rai::Configuration>();
  // rai::Skeleton::getTwoWaypointProblem(t, *C, q0, qT, *ways->komoWaypoints);
  ways->komoWaypoints->getSubProblem(t, *C, q0, qT);

  //cout <<C.getJointNames() <<endl;

  for(rai::Frame* f:C->frames) f->ensure_X();
  rrt = make_shared<RRT_PathFinder>();
  rrt->P = make_shared<ConfigurationProblem>(C, true, root->info->rrtTolerance);
  if(root->tamp.explicitCollisions.N) rrt->P->setExplicitCollisionPairs(root->tamp.explicitCollisions);
  rrt->P->useBroadCollisions = root->tamp.useBroadCollisions;
  rrt->setStartGoal(q0, qT);
  rrt->opt.stepsize = root->info->rrtStepsize;

  if(root->verbose()>1) rrt->opt.verbose=root->verbose()-2;
  rrt->opt.maxIters = root->info->rrtStopEvals;
}

rai::TaskPlan rai::LGPComp2_RRTpath::getTaskPlan() {
  if(!taskPlan.empty)  return taskPlan;
  taskPlan = TaskPlan(sket->actionSequence);
  rai::Array<Action> relevant_action = {taskPlan.actions(t)};
  taskPlan = TaskPlan(relevant_action);
  return taskPlan;
}

double rai::LGPComp2_RRTpath::branchingPenalty_child(int i) {
  return 0.0;
}

void rai::LGPComp2_RRTpath::untimedCompute() {
  LGPComp2_root* root = ways->sket->root;

  int r=0;
  double quantum = root->info->quantum;
  auto startTime = rai::cpuTime();
  while(rai::cpuTime() - startTime < quantum) { r = rrt->stepConnect(); if(r) break; }
  if(r==1) {
    // cout << "RRT succeeded" <<endl;
    isComplete=true;
    l=0.;
    path = path_resampleLinear(rrt->path, root->info->pathStepsPerPhase);
  }
  if(r==-1) {
    isComplete=true;
    l=1e10;
    //      if(root->sol->opt.verbose>0) komoPath->view(root->sol->opt.verbose>1, "init path - RRT FAILED");
    if(root->verbose()>1) LOG(-1) <<"RRT FAILED";
    // cout << "RRT FAILED" <<endl;
    path.clear();
  }
  if(isComplete) {
    rrt.reset();
  }
}

std::shared_ptr<rai::ComputeNode> rai::LGPComp2_RRTpath::createNewChild(int i) {
  // CHECK(!i, "only single child");
  // cout << "creating child from RRTpath at t=" << t << "node ID:" << ID<< endl;
  if(t+1 < ways->komoWaypoints->T) {
    auto rrt =  make_shared<LGPComp2_RRTpath>(this, ways, t+1, i);
    rrt->prev = this;
    return rrt;
  }
  return make_shared<LGPComp2_OptimizePath>(this, ways, i);
}

//===========================================================================

rai::LGPComp2_OptimizePath::LGPComp2_OptimizePath(rai::LGPComp2_Skeleton* _sket)
  : GittinsNode(_sket), sket(_sket) {
  name <<"LGPComp2_PathFromSket#"<<sket->num;

  LGPComp2_root* root=sket->root;

  isTerminal = true;

  komoPath = root->tamp.get_fullMotionProblem(root->C, sket->actionSequence, {});
  // komoPath->clone(*sket->skeleton.komoPath);

  //random initialize
  komoPath->initRandom(0);
  if(root->verbose()>2) komoPath->view(root->verbose()>3, STRING(name <<" - init random from Skeleton directly"));
  if(root->verbose()>3) komoPath->view_play(true, 0, .1);

  komoPath->run_prepare(0.);
  //  komoPath->opt.animateOptimization=2;
  sol.setProblem(komoPath->nlp());
  sol.setInitialization(komoPath->x);
  //sol.setOptions(OptOptions().set_verbose(4));
}

rai::LGPComp2_OptimizePath::LGPComp2_OptimizePath(rai::LGPComp2_Waypoints* _ways)
  : GittinsNode(_ways), sket(_ways->sket), ways(_ways) {
  name <<"LGPComp2_PathFromWay#"<<ways->seed;

  LGPComp2_root* root=sket->root;

  isTerminal = true;

  komoPath = root->tamp.get_fullMotionProblem(root->C, sket->actionSequence, ways->komoWaypoints);
  // komoPath = make_shared<KOMO>();
  // komoPath->clone(*sket->skeleton.komoPath);

  //initialize by waypoints
  komoPath->initWithWaypoints(ways->komoWaypoints->getPath_qAll(), 1, true);
  komoPath->run_prepare(0.);
  if(root->verbose()>2) komoPath->view(root->verbose()>3, STRING(name <<" - init with interpolated waypoints"));
  if(root->verbose()>3) komoPath->view_play(true, 0, .1);

  komoPath->run_prepare(0.);
  //  komoPath->opt.animateOptimization=2;
  sol.setProblem(komoPath->nlp());
  sol.setInitialization(komoPath->x);
  //sol.setOptions(OptOptions().set_verbose(4));
}

rai::LGPComp2_OptimizePath::LGPComp2_OptimizePath(rai::LGPComp2_RRTpath* _par, rai::LGPComp2_Waypoints* _ways, int rndSeed)
  : GittinsNode(_par), sket(_ways->sket), ways(_ways), seed(rndSeed) {
  name <<"LGPComp2_PathFromRRT#"<<ways->seed;
  taskPlan = TaskPlan(sket->actionSequence);
  rnd.seed(rndSeed);
  LGPComp2_root* root = ways->sket->root;

  isTerminal = true;

  komoPath = root->tamp.get_fullMotionProblem(root->C, sket->actionSequence, ways->komoWaypoints);

  //collect path and initialize
  rai::Array<LGPComp2_RRTpath*> rrts(ways->komoWaypoints->T);
  rrts=0;
  rrts(-1) = _par;
  for(uint t=ways->komoWaypoints->T-1; t--;) {
    rrts(t) = rrts(t+1)->prev;
    CHECK_EQ(rrts(t)->t, t, "");
  }

  //first set waypoints (including action parameters!)
  komoPath->initWithWaypoints(ways->komoWaypoints->getPath_qAll());   //non-interpolating
//  if(rrtpath.N) komoPath->initWithPath_qOrg(rrtpath);
  komoPath->run_prepare(0.);
  if(root->verbose()>2) komoPath->view(root->verbose()>3, STRING(name <<" - init with constant waypoints"));
  if(root->verbose()>3) komoPath->view_play(true, 0, .1);

  for(uint t=0; t<ways->komoWaypoints->T; t++) {
    CHECK(rrts(t)->isFeasible, "rrt of t=" <<t <<" is infeasible - can't use RRT-initialized KOMO")
    komoPath->initPhaseWithDofsPath(t, rrts(t)->C->getDofIDs(), rrts(t)->path, false);
    if(root->verbose()>2) {
      komoPath->view(root->verbose()>3, STRING(name <<" - init with rrt part" <<t));
      rai::wait(.1);
    }
  }

  if(root->verbose()>1) komoPath->view(root->verbose()>3, STRING(name <<" - init with rrts"));

  komoPath->run_prepare(0.);
  //  komoPath->opt.animateOptimization=2;
  sol.setProblem(komoPath->nlp());
  sol.setInitialization(komoPath->x);
  //sol.setOptions(OptOptions().set_verbose(4));
}

rai::TaskPlan rai::LGPComp2_OptimizePath::getTaskPlan() {
  if(!taskPlan.empty)  return taskPlan;
  taskPlan = TaskPlan(sket->actionSequence);
  return taskPlan;
}

void rai::LGPComp2_OptimizePath::untimedCompute() {
  for(uint i=0; i<5; i++) if(sol.step()) break;

  LGPComp2_root* root = sket->root;

  l = sol.ret->eq + sol.ret->ineq;
  isComplete = sol.ret->done;
  sol.ret->feasible = (sol.ret->ineq + sol.ret->eq < 3.);

  if(!isComplete && root->verbose()>2) {
    komoPath->pathConfig.get_viewer()->renderUntil=rai::_shadow;
    komoPath->view(root->verbose()>5, STRING(name <<" - intermediate result c:" <<c <<"\n" <<*sol.ret));
    if(root->verbose()>3) {
      if(root->verbose()>5) komoPath->view_play(true, 0, .1);
      else komoPath->view_play(false, 0, .1);
    }
  }

  if(isComplete) {
    if(root->verbose()>0 && ways) ways->komoWaypoints->view_close();
    if(root->verbose()>0) komoPath->pathConfig.get_viewer()->renderUntil=rai::_shadow;

    if(root->verbose()>0) LOG(0) <<"path " <<*sol.ret;
    if(root->verbose()>1) cout <<komoPath->report(false, true, root->verbose()>2);
    if(root->verbose()>0) komoPath->view(root->verbose()>2, STRING(name <<" - optimized \n" <<*sol.ret <<"\n" <<sket->actionSequence <<"\n"));
    //komoPath->checkGradients();
    if(root->verbose()>1) komoPath->view_play(root->verbose()>1 && sol.ret->feasible);

    if(!sol.ret->feasible) {
      //l = 1e10;
      isFeasible = false;
      komoPath->view_close();
      if(root->verbose()>1) {
        cout <<sol.optCon->L.reportGradients(komoPath->featureNames) <<endl;
      }
    } else {
      isFeasible = true;
      if(root->verbose()>0) { //save video and path and everything
        auto path = STRING("z.sol_"<<ID<<"/");

        komoPath->view_play(false, 0, .1, path);
        ofstream fil(path + "info.txt");
        fil <<*sol.ret <<"\n\nSkeleton:{" <<sket->actionSequence <<"\n}" <<endl;
        fil <<komoPath->report() <<endl;
        fil <<sol.optCon->L.reportGradients(komoPath->featureNames) <<endl;
        ofstream cfil(path + "last.g");
        komoPath->world.setFrameState(komoPath->getConfiguration_X(komoPath->T-1));
        cfil <<komoPath->world;
      }
    }
    if(root->verbose()<2) komoPath->view_close();
  }

  if(isComplete) {
    //if(ways) ways->komoWaypoints.reset();
    komoPath.reset();
  }
}
