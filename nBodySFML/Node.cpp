#include "Node.h"
#include <memory>
#include <thread>

#define _MAX_DEPTH 100

int thread_count = std::thread::hardware_concurrency();

Node::Node(unsigned int pDepth)
{
	HasChildren = false;
	Depth = pDepth;
}

Node::Node()
{
	HasChildren = false;
	Depth = 0;
}

void Node::GenerateChildren()
{
	std::vector<Body*> q1Bodies, q2Bodies, q3Bodies, q4Bodies;

#pragma omp parallel for num_threads(thread_count) private(q1Bodies, q2Bodies, q3Bodies, q4Bodies)
	for (int i = 0; i < Bodies.size(); i++)
	{
		if (Bodies[i]->posX < (posX + (width / 2)))     //if true, 1st or 3rd
		{
			if (Bodies[i]->posY < (posY + (height / 2)))    //1
			{
				q1Bodies.push_back(Bodies[i]);
			}
			else //3
			{
				q3Bodies.push_back(Bodies[i]);
			}
		}
		else                                            //2 or 4
		{
			if (Bodies[i]->posY < (posY + (height / 2)))    //2
			{
				q2Bodies.push_back(Bodies[i]);
			}
			else //4
			{
				q4Bodies.push_back(Bodies[i]);
			}
		}
	}

	Node* q1 = new Node(Depth + 1);
	Node* q2 = new Node(Depth + 1);
	Node* q3 = new Node(Depth + 1);
	Node* q4 = new Node(Depth + 1);

	q1->SetParam(q1Bodies, width / 2, height / 2, posX, posY);
	q2->SetParam(q2Bodies, width / 2, height / 2, posX + (width / 2), posY);
	q3->SetParam(q3Bodies, width / 2, height / 2, posX, posY + (height / 2));
	q4->SetParam(q4Bodies, width / 2, height / 2, posX + (width / 2), posY + (height / 2));

	Child.push_back(q1);
	Child.push_back(q2);
	Child.push_back(q3);
	Child.push_back(q4);

	HasChildren = true;
}

void Node::SetParam(std::vector<Body*> pBodies, float pwidth, float pheight, float px, float py)
{
	Bodies = pBodies;
	posX = px;
	posY = py;
	width = pwidth;
	height = pheight;

	float mass = 0;
	double Centerx = 0;
	double Centery = 0;

#pragma omp parallel for num_threads(thread_count) schedule(static, 1)
	for (int i = 0; i < pBodies.size(); i++)
	{
		mass += pBodies[i]->mass;
		Centerx += pBodies[i]->posX; 
		Centery += pBodies[i]->posY;
	}

	TotalMass = mass;

	unsigned int size = pBodies.size();

	CenterOfMassx = Centerx / size;
	CenterOfMassy = Centery / size;

	if (Bodies.size() > 1 && Depth < _MAX_DEPTH)
	{
		GenerateChildren();
	}
}

void Node::Reset()
{
	Bodies.clear();

	for (unsigned int i = 0; i < Child.size(); i++)
	{
		Child[i]->Reset();
	}

	for (unsigned int i = 0; i < Child.size(); i++)
	{
		delete Child[i];
	}

	Child.clear();

	HasChildren = false;
}

Node::~Node()
{

}
