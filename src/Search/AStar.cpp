/*  ------------------------------------------------------------------
    Copyright (c) 2011-2024 Marc Toussaint
    email: toussaint@tu-berlin.de

    This code is distributed under the MIT License.
    Please see <root-path>/LICENSE for details.
    --------------------------------------------------------------  */

#include "AStar.h"

#include <fstream>
#include <iomanip>
#include <typeinfo>
#include "../Core/util.h"         // niceTypeidName(...)
#include "Search/ComputeNode.h"          // for ComputeNode::c
#include "../LGP/LGP_computers.h"
#include "Search/GittinsNode.h" // for GittinsNode
#include <Search/NodeTypes.h>

static void logNodeCompletion(rai::TreeSearchNode* node){
  using namespace rai;
  auto comp = dynamic_cast<GittinsNode*>(node);
  if(!comp) return; // only log ComputeNode-based nodes
  NodeType thisType = comp->getNodeType();
  if(thisType == NodeType::Other || thisType == NodeType::Skeleton) return; // skip Other type nodes


  const char* path = "completionsAroundProblem1.csv";
  std::ifstream in(path);
  bool exists = in.good();
  in.close();

  std::ofstream out(path, std::ios::app);
  if(!exists){
    out << "c,feasible,type,plan\n";
  }

  out << comp->c << ','
      << (node->isFeasible ? 1 : 0) << ','
      << '"' << rai::toString(thisType) << '"' << ','
      << '"' << comp->getTaskPlan() << '"' << '\n';
}

rai::AStar::AStar(const std::shared_ptr<rai::TreeSearchNode>& _root, SearchMode _mode)
  : root(_root), mode(_mode) {
  root->ID = 0;
  mem.append(root);
  addToQueue(root.get());
}

void rai::AStar::step() {
  if(mode != DataExtraction) {
    stepAStar();
    return;
  }
  steps++;
  bool initialized = false;
  std::vector<GittinsNode*> feasibleSkeletons;
  std::vector<GittinsNode*> feasibleWaypoints;
  std::vector<GittinsNode*> rrtPath;
  int skeletonCount = 0;
  int waypointCount = 0;

  rai::TreeSearchNode* root = queue.pop();
  if (!initialized) {
    for (int i = 0; i < 50 && skeletonCount < 30; i++) {
      std::cout << "Generating skeleton child " << i << " of " << 50 << std::endl;
      auto child = root->transition(i);
      if (child) {
        child->ID = mem.N;
        mem.append(child);
        auto gittinsChild = dynamic_cast<GittinsNode*>(child.get());
        if (gittinsChild) {
          while(!gittinsChild->isComplete) {
            gittinsChild->compute();
          }
        }

        if (child->isFeasible && 
            gittinsChild->getNodeType() == rai::NodeType::Skeleton) {
          feasibleSkeletons.push_back(gittinsChild);
          skeletonCount++;
        }
      }
    }
    initialized = true;
    std::cout << "Generated " << skeletonCount << " feasible skeleton nodes" << std::endl;
  }
  
  for (size_t skel_idx = 0; skel_idx < feasibleSkeletons.size(); skel_idx++) {
    auto skeleton = feasibleSkeletons[skel_idx];
    std::cout << "Processing skeleton " << (skel_idx + 1) << "/" << feasibleSkeletons.size() << std::endl;
    
    for (int i = 0; i < 20; i++) {
      if (i % 10 == 0) {
        std::cout << "  Generating waypoint " << i << "/10 for skeleton " << (skel_idx + 1) << std::endl;
      }
      auto waypointChild = skeleton->transition(i);
      if (waypointChild) {
        waypointChild->ID = mem.N;
        mem.append(waypointChild);

        while(!waypointChild->isComplete) {
          waypointChild->compute();
        }
        logNodeCompletion(waypointChild.get());
        
        // Check if it's a feasible waypoint node
        auto gittinsWaypoint = dynamic_cast<GittinsNode*>(waypointChild.get());
        if (gittinsWaypoint && waypointChild->isFeasible && 
            gittinsWaypoint->getNodeType() == rai::NodeType::WaypointsNode) {
          feasibleWaypoints.push_back(gittinsWaypoint);
          waypointCount++;
        }
      }
    }
  }
  
  std::cout << "Generated " << waypointCount << " feasible waypoint nodes" << std::endl;
  
  for (size_t wp_idx = 0; wp_idx < feasibleWaypoints.size(); wp_idx++) {
    auto waypoint = feasibleWaypoints[wp_idx];
    std::cout << "Processing waypoint " << (wp_idx + 1) << "/" << feasibleWaypoints.size() << std::endl;
    
    TaskPlan taskPlan = waypoint->getTaskPlan();
    int numActions = taskPlan.actions.N;

    // Try exactly 1 time to compute the full RRT path
    for (int attempt = 0; attempt < 1; attempt++) {
      std::cout << "  RRT attempt " << (attempt + 1) << "/1 for waypoint " << (wp_idx + 1) << std::endl;

      // Start from the waypoint and traverse through all RRT nodes for each action
      GittinsNode* currentNode = waypoint;
      bool fullPathFeasible = true;
      
      // For each action in the task plan, compute the corresponding RRT node
      for (int action = 0; action < numActions && fullPathFeasible; action++) {
        std::cout << "    Processing action " << (action + 1) << "/" << numActions << std::endl;
        
        auto rrtChild = currentNode->transition(attempt); // Use attempt as variation
        if (rrtChild) {
          rrtChild->ID = mem.N;
          mem.append(rrtChild);
          
          // Compute the RRT node
          while (!rrtChild->isComplete) {
            rrtChild->compute();
          }
          
          logNodeCompletion(rrtChild.get());
          
          // Check if it's a feasible RRT node
          auto gittinsRRT = dynamic_cast<GittinsNode*>(rrtChild.get());
          if (gittinsRRT && rrtChild->isFeasible && 
              gittinsRRT->getNodeType() == rai::NodeType::RRTNode) {
              currentNode = gittinsRRT; // Move to next RRT node for next action
              if(action == numActions - 1) rrtPath.push_back(gittinsRRT);
          } else {
            fullPathFeasible = false;
          }
        } else {
          fullPathFeasible = false;
          std::cout << "    Could not create RRT node for action " << (action + 1) << std::endl;
        }
      }
    }
  }
  for (auto& lastRRTNode : rrtPath) {
    // Try exactly 30 times to compute LGP path from the last RRT node
    for (int lgpAttempt = 0; lgpAttempt < 100; lgpAttempt++) {
      std::cout << "      LGP attempt " << (lgpAttempt + 1) << "/30" << std::endl;

      auto lgpChild = lastRRTNode->transition(lgpAttempt);
      if (lgpChild) {
        lgpChild->ID = mem.N;
        mem.append(lgpChild);
        
        // Compute the LGP path node
        while (!lgpChild->isComplete) {
          lgpChild->compute();
        }
        
        logNodeCompletion(lgpChild.get());
        
        // Check if it's a feasible LGP path node
        auto gittinsLGP = dynamic_cast<GittinsNode*>(lgpChild.get());
        if (gittinsLGP && lgpChild->isFeasible && 
            gittinsLGP->getNodeType() == rai::NodeType::LGPPathNode) {
          solutions.append(lgpChild.get());
        }
      }
    }
  }
}

void rai::AStar::stepAStar() {
  steps++;

  //pop
  TreeSearchNode* node = 0;
  if(mode==astar || mode==FIFO){
    if(!queue.N) {
      LOG(-1) <<"AStar: queue is empty -> failure!";
      return;
    }
    // printFrontier();
    node = queue.pop();
    // if(mode==astar){
    //   CHECK_GE(node->f_prio, currentLevel, "level needs to increase");
    // }
    currentLevel = node->f_prio;
  }else if(mode==treePolicy){
    node = selectByTreePolicy();
  }
  //    LOG(0) <<"looking at node '" <<*node <<"'";

  //widen
  TreeSearchNode *siblingToBeAdded = 0;
  if(node->needsWidening){
    CHECK(node->parent, "");
    NodeP sibling = node->parent->transition(node->parent->children.N);
    if(sibling){
      CHECK_EQ(sibling->parent, node->parent, "")
      CHECK_GE(sibling->f_prio, currentLevel, "sibling needs to have greater level")
      sibling->ID = mem.N;
      mem.append(sibling);
      siblingToBeAdded = sibling.get();
      //queue.add(sibling->f_prio, sibling.get(), false);
      if(node->parent->getNumDecisions()==-1) sibling->needsWidening=true;
    }
    node->needsWidening=false;
  }

  //compute
  bool wasComplete = node->isComplete;
  if(!node->isComplete){
    node->compute();
  }

  bool becameComplete = (!wasComplete && node->isComplete);
  if(becameComplete){
    // logNodeCompletion(node);
    // printFrontier();
    // std::cout <<" node '" <<*node <<"' became complete and feas: "<< node->isFeasible <<std::endl;
  }

  //depending on state -> drop, reinsert, save as solution, or expand
  if(!node->isFeasible){ //drop node completely

  }else if(!node->isComplete){ //send back to queue
    addToQueue(node);

  }else if(mode==astar && node->f_prio>currentLevel){ //send back to queue - might not be optimal anymore
    addToQueue(node);

  }else if(node->isTerminal){   //save as solution
    // if the node can be casted to a gittinsNode, cast it and print the taskPlan
    // if(auto gittinsNode = dynamic_cast<GittinsNode*>(node)){
    //   std::cout << "GittinsNode found with TaskPlan: " << gittinsNode->taskPlan << std::endl;
    // }
    solutions.append(node);

  }else{  //expand or deepen
    CHECK(node->isComplete, "");
    CHECK(!node->isTerminal, "");

    //LOG(0) <<"expanding node '" <<*node <<"'";
    int n = node->getNumDecisions();
    uint createN = n;
    if(n==-1){ createN=1; } //infinity -> add only the first

    for(uint i=0;i<createN;i++) {
      NodeP child = node->transition(i);
      // CHECK_EQ(child->parent, node, "")
          // CHECK_GE(child->f_prio, currentLevel, "children needs to have greater level")
      child->ID = mem.N;
      mem.append(child);
      //child->compute();
      //if(!node->isFeasible) return false;
      addToQueue(child.get());
      if(n==-1) child->needsWidening=true;
    }

  }

  //remember inserting the sibling, FIFO style (also to allow for FIFO mode)
  if(siblingToBeAdded){
    addToQueue(siblingToBeAdded);
  }
}

bool rai::AStar::run(int stepsLimit) {
  uint numSol=solutions.N;
  for(;;) {
    step();
    if(solutions.N>numSol) break;
    //      report();
    if(stepsLimit>=0 && (int)steps>=stepsLimit) break;
  }
  if(verbose>0){
    LOG(0) <<"==== DONE ===";
    report();
  }
  if(solutions.N>numSol) return true;
  return false;
}

void rai::AStar::report(){
  std::cout <<" iters: " <<steps
           <<" mem#: " <<mem.N
          <<" queue#: " <<queue.N <<endl;
  if(verbose>2) std::cout <<" queue: " <<queue <<std::endl;
  if(solutions.N) std::cout <<" solutions: " <<solutions.modList();
}

void rai::AStar::printFrontier() const {
  std::cout << "=== Frontier (Queue) Contents ===" << std::endl;
  std::cout << "Queue size: " << queue.N << std::endl;
  
  if(queue.N == 0) {
    std::cout << "Queue is empty." << std::endl;
    return;
  }
  
  // Create a copy of the queue to iterate through without modifying the original
  auto queueCopy = queue;
  
  std::cout << "Priority | Node Name | Additional Info" << std::endl;
  std::cout << "---------|-----------|----------------" << std::endl;
  
  while(queueCopy.N > 0) {
    TreeSearchNode* node = queueCopy.pop();
    
    std::cout << std::setw(8) << std::fixed << std::setprecision(3) << node->f_prio 
              << " | " << std::setw(9) << node->name();
    
    // Check if the node is a GittinsNode and print its TaskPlan
    auto gittinsNode = dynamic_cast<GittinsNode*>(node);
    if(gittinsNode && !gittinsNode->taskPlan.empty) {
      std::cout << " | TaskPlan: " << gittinsNode->taskPlan.toString();
    } else {
      std::cout << " | (no TaskPlan)";
    }
    
    std::cout << std::endl;
  }
  
  std::cout << "=================================" << std::endl;
}

rai::TreeSearchNode* rai::AStar::selectByTreePolicy(){
  rai::TreeSearchNode *node = root.get();

  //-- TREE POLICY
  while(node->children.N
        //(int)node->children.N == node->getNumDecisions()  //# we are 'inside' the full expanded tree: children for each action -> UCB to select the most promising
        && !node->isTerminal){                       //# we're not at a terminal yet

    // compute the UCB scores for all children
    arr scores(node->children.N);
    for(uint i=0;i<scores.N;i++) scores(i) = node->children(i)->treePolicyScore(i);
    //child->data_Q / child->data_n + beta * sqrt(2. * ::log(node->data_n)/child->data_n);

    // pick the child with highest
    node = node->children(argmax(scores));
  }

  return node;
}

void rai::AStar::addToQueue(TreeSearchNode *node){
  if(mode==FIFO) queue.append(node);
  else queue.add(node->f_prio, node, true);
}
