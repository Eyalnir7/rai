/*  ------------------------------------------------------------------
    Copyright (c) 2011-2024 Marc Toussaint
    email: toussaint@tu-berlin.de

    This code is distributed under the MIT License.
    Please see <root-path>/LICENSE for details.
    --------------------------------------------------------------  */

#include "GittinsSearch.h"

#include <fstream>
#include <iomanip>
#include <typeinfo>
#include "../Core/util.h"          // niceTypeidName(...)
#include "../Search/ComputeNode.h" // for ComputeNode::c
#include "../Search/GittinsNode.h" // for GittinsNode
#include "../Search/NodeTypes.h"
#include "LGP_computers2.h"

rai::GittinsSearch::GittinsSearch(const std::shared_ptr<LGPComp2_root> &_root)
    : root(_root)
{
    root->ID = 0;
    // mem.append(root);
    // addToQueue(root.get());
}

void rai::GittinsSearch::step()
{
    steps++;
    // pop
    if (root->numSkeletonsTried == root->children.N)
    {
        if (opt.verbose >= 1)
            cout << "Adding new skeletons to root node" << endl;
        for (int i = 0; i < opt.numTaskPlans; i++)
        {
            NodeP skeletonNode = root->transitionToGittinsNode(root->children.N);
            skeletonNode->ID = mem.N;
            mem.append(skeletonNode);
            addToQueue(skeletonNode.get());
        }
    }
    GittinsNode *node = 0;
    if (!queue.N)
    {
        LOG(-1) << "AStar: queue is empty -> failure!";
        return;
    }
        // printFrontier();
    node = queue.pop();
    currentLevel = node->f_prio;
    //    LOG(0) <<"looking at node '" <<*node <<"'";

    // widen
    GittinsNode *siblingToBeAdded = 0;
    if (node->needsWidening)
    {
        CHECK(node->parent, "");
        NodeP sibling = node->getGittinsParent()->transitionToGittinsNode(node->parent->children.N);
        if (sibling)
        {
            CHECK_EQ(sibling->parent, node->parent, "")
            CHECK_GE(sibling->f_prio, currentLevel, "sibling needs to have greater level")
            sibling->ID = mem.N;
            mem.append(sibling);
            siblingToBeAdded = sibling.get();
            // queue.add(sibling->f_prio, sibling.get(), false);
            if (node->parent->getNumDecisions() == -1)
                sibling->needsWidening = true;
        }
        node->needsWidening = false;
    }

    // compute
    bool wasComplete = node->isComplete;
    if (!node->isComplete)
    {
        if (opt.verbose >= 2)
            cout << "invested compute in " << *node << "child of" << *node->parent << endl;
        node->compute();
    }

    // depending on state -> drop, reinsert, save as solution, or expand
    if (!node->isFeasible)
    { // drop node completely
    }
    else if (!node->isComplete)
    { // send back to queue
        addToQueue(node);

        // }else if(mode==astar && node->f_prio>currentLevel){ //send back to queue - might not be optimal anymore
        //     addToQueue(node);
    }
    else if (node->isTerminal)
    { // save as solution
        // if the node can be casted to a gittinsNode, cast it and print the taskPlan
        // if(auto gittinsNode = dynamic_cast<GittinsNode*>(node)){
        //   std::cout << "GittinsNode found with TaskPlan: " << gittinsNode->taskPlan << std::endl;
        // }
        solutions.append(node);
    }

    else
    { // expand or deepen
        CHECK(node->isComplete, "");
        CHECK(!node->isTerminal, "");

        // LOG(0) <<"expanding node '" <<*node <<"'";
        int n = node->getNumDecisions();
        uint createN = n;
        if (n == -1)
        {
            createN = 1;
        } // infinity -> add only the first

        for (uint i = 0; i < createN; i++)
        {
            NodeP child = node->transitionToGittinsNode(i);
            // CHECK_EQ(child->parent, node, "")
            // CHECK_GE(child->f_prio, currentLevel, "children needs to have greater level")
            child->ID = mem.N;
            mem.append(child);
            // child->compute();
            // if(!node->isFeasible) return false;
            addToQueue(child.get());
            if (n == -1)
                child->needsWidening = true;
        }
    }

    // remember inserting the sibling, FIFO style (also to allow for FIFO mode)
    if (siblingToBeAdded)
    {
        addToQueue(siblingToBeAdded);
    }

    bool becameComplete = (!wasComplete && node->isComplete);
    if (becameComplete && opt.verbose > 1)
    {
        printFrontier();
        std::cout << " node '" << *node << "' became complete and feas: " << node->isFeasible << " after: " << node->c << std::endl;
    }
}

bool rai::GittinsSearch::run(int stepsLimit)
{
    uint numSol = solutions.N;
    for (;;)
    {
        step();
        if (solutions.N > numSol)
            break;
        //      report();
        if (stepsLimit >= 0 && (int)steps >= stepsLimit)
            break;
    }
    if (verbose > 0)
    {
        LOG(0) << "==== DONE ===";
        report();
    }
    if (solutions.N > numSol)
        return true;
    return false;
}

void rai::GittinsSearch::report()
{
    std::cout << " iters: " << steps
              << " mem#: " << mem.N
              << " queue#: " << queue.N << endl;
    if (verbose > 2)
        std::cout << " queue: " << queue << std::endl;
    if (solutions.N)
        std::cout << " solutions: " << solutions.modList();
}

void rai::GittinsSearch::printFrontier() const
{
    std::cout << "=== Frontier (Queue) Contents ===" << std::endl;
    std::cout << "Queue size: " << queue.N << std::endl;

    if (queue.N == 0)
    {
        std::cout << "Queue is empty." << std::endl;
        return;
    }

    // Create a copy of the queue to iterate through without modifying the original
    auto queueCopy = queue;

    std::cout << "Priority | Node Name | Additional Info" << std::endl;
    std::cout << "---------|-----------|----------------" << std::endl;

    bool seen_waypoints = false;
    while (queueCopy.N > 0)
    {
        TreeSearchNode *node = queueCopy.pop();

        // Only print RRT or LGP nodes, and at most one waypoints node
        auto gittinsNode = dynamic_cast<GittinsNode *>(node);
        if (gittinsNode)
        {
            NodeType nodeType = gittinsNode->getNodeType();
            // Skip if it's a waypoints node and we've already seen one
            if (nodeType != NodeType::RRTNode && nodeType != NodeType::LGPPathNode)
            {
                if (seen_waypoints)
                {
                    continue;
                }
                seen_waypoints = true;
            }
        }
        else
        {
            continue;
        }

        std::cout << std::setw(8) << std::fixed << std::setprecision(3) << node->f_prio
                  << " | " << std::setw(9) << node->name();

        // Check if the node is a GittinsNode and print its TaskPlan
        if (gittinsNode && !gittinsNode->taskPlan.empty)
        {
            std::cout << " | parent: " << *gittinsNode->parent << " | compute invested: " << gittinsNode->c;
        }
        else
        {
            std::cout << " | (no TaskPlan)";
        }

        std::cout << std::endl;
    }

    std::cout << "=================================" << std::endl;
}

void rai::GittinsSearch::addToQueue(GittinsNode *node)
{
    queue.add(node->f_prio, node, true);
}
