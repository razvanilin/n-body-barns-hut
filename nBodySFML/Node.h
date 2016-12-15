#pragma once
#include <vector>

/*struct Body
{
	float posX, posY;           //position x and y
	float velX, velY;           //velocity x and y
	double forceX, forceY;      //force acting on object since last frame
	float mass;                 //mass of object
};*/

class Body
{
public:
	Body(float px, float py, float pmass, float pvx = 0, float pvy = 0) :
		posX(px),
		posY(py),
		velX(pvx),
		velY(pvy),
		forceX(0),
		forceY(0),
		mass(pmass) {}

	float posX, posY;           //position x and y
	float velX, velY;           //velocity x and y
	double forceX, forceY;      //force acting on object since last frame
	float mass;		//mass of object
};

class Node
{
public:
	std::vector<Body*> Bodies;
	std::vector<Node*> Child;
	bool HasChildren;

	float posX, posY;
	float width, height;
	float TotalMass;
	float CenterOfMassx;
	float CenterOfMassy;
	unsigned int Depth;
	bool IsUsed;            //For testing, delete this later

	Node();
	Node(unsigned int pdepth);
	~Node();
	void GenerateChildren();
	void SetParam(std::vector<Body*> pBodies, float pwidth, float pheight, float px = 0, float py = 0);
	void Reset();
};

