using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using UnityEngine.UI;
using System.Linq;
using Newtonsoft.Json;
using System.Net;
using System.Net.Sockets;
using System.Text;

public class CombatAgent : Agent
{
    [Header("Specific to GridWorld")]
    private CombatAcademy academy;
    public float timeBetweenDecisionsAtInference;
    private float timeSinceDecision;

    float actionSucc = 1f;

   
    public static Vector3[] actionPos;

    public GameObject bullet = null;

    public LineRenderer hpBar;

    public float moveSpeed = 2;

    Vector3 preRewordPos;
    Vector3 prePos;

    float health;


    static float rotationAngle = 10f;




    public override void InitializeAgent()
    {
        academy = FindObjectOfType(typeof(CombatAcademy)) as CombatAcademy;

        if (actionPos == null)
        {
            actionPos = new Vector3[5];


            actionPos[0].Set(0f, 0, 1f);
            actionPos[1].Set(0f, 0, -1f);
            actionPos[2].Set(-1f, 0, 0f);
            actionPos[3].Set(1f, 0, 0f);
            actionPos[4].Set(0, 0, 0f);

        }

        preRewordPos = transform.position;


        Init();
    }

    public void Init()
    {
        preRewordPos = transform.position;

        health = 5;
    }


    public bool SetDamage()
    {
        if (health<1)
            return false;

      
        for(int i=0;i< hpBar.positionCount;++i)
        {
            Vector3 pos = hpBar.GetPosition(i);
            hpBar.SetPosition(i, new Vector3(pos.x - (pos.x / health), 0, 0));
        }
        health -= 1;

        return true;

    }

   

    public override void CollectObservations()
    {

        if (brain == null)
            return;

        AddVectorObs(gameObject.transform.position.x);
        AddVectorObs(gameObject.transform.position.z);
      
   

    }

    public void RuleBaseAction()
    {
        float xMinDist = 10000;
        GameObject nearObj = null;


 
   

        for (int i=0; i < academy.trueAgent.Length;++i)
        {

            if(academy.trueAgent[i].GetComponent<CombatAgent>().health<1)
            {

                continue;
            }

            float xDist = Vector3.SqrMagnitude(transform.position - academy.trueAgent[i].transform.position);

            if(xMinDist>xDist)
            {
                xMinDist = xDist;
                nearObj = academy.trueAgent[i];
            }
        }

        if(nearObj ==null)
        {
            academy.Done();
            return;
        }

        Vector3 dir = nearObj.transform.position - transform.position;


        System.Action<Vector3> MovePos = (Vector3 v) =>
        {
            float max = 0;
            int d = -1;

            for (int j = 0; j < actionPos.Length; ++j)
            {
                float dv = Vector3.Dot(v, actionPos[j]);
                if (max < dv)
                {
                    max = dv;
                    d = j;
                }
            }

            Vector3 targetPos = transform.position;

            targetPos = transform.position + actionPos[d];

            if (MoveAble(ref targetPos))
            {
                transform.position = targetPos;
            }
        };



        if (xMinDist<30)
        {


            Quaternion q = Quaternion.FromToRotation(Vector3.zero, dir);

    
            float angleRest = Mathf.Abs(q.eulerAngles.y % rotationAngle);


            if(angleRest>3&&angleRest<7)  ///이동해야 한다..
            {

                MovePos(dir);

            }
            else
            {
                float angleDiff = q.eulerAngles.y - transform.rotation.eulerAngles.y;

                if (angleDiff > 180)
                    angleDiff = angleDiff - 360;


                if (Mathf.Abs(angleDiff)< 5|| Mathf.Abs(angleDiff)>354)
                {
                    Shot();
                }
                else
                {
                    if(angleDiff>0)
                    {
                        Turn(true);
                    }
                    else
                    {
                        Turn(false);
                    }
                }

            }

            return;
        }

        MovePos(dir);
    }

    public bool MoveAble(ref Vector3 pos)
    {      
        Collider[] blockTest = Physics.OverlapBox(pos, new Vector3(0.3f, 0.3f, 0.3f));


        if (blockTest.Where(col => col.gameObject.tag == "wall" || col.gameObject.tag == "agent" || col.gameObject.tag == "enermy").ToArray().Length == 0)
        {
            return true;
        }

        return false;
    }

    // to be implemented by the developer
    public override void AgentAction(float[] vectorAction, string textAction)
    {

        if (brain == null)
            return;

        actionSucc = 0;
        AddReward(-0.001f);

        if (health < 1)
            return;


        if (GetStepCount() >= agentParameters.maxStep)
        {
            academy.WriteSummary(GetStepCount());
            Done();
            return;
        }


        Vector3 dirToGo = Vector3.zero;
        Vector3 rotateDir = Vector3.zero;


        bool shootCommand = false;
        bool shoot = false;



        if (brain.brainParameters.vectorActionSpaceType == SpaceType.continuous)
        {
            dirToGo = transform.forward * Mathf.Clamp(vectorAction[0], -1f, 1f);
            rotateDir = transform.up * Mathf.Clamp(vectorAction[1], -1f, 1f);
            shootCommand = Mathf.Clamp(vectorAction[2], 0f, 1f) > 0.5f;
        }
        else
        {
            switch ((int)(vectorAction[0]))
            {
                case 1:
                    dirToGo = transform.forward;
                    break;
                case 2:
                    shootCommand = true;
                    break;
                case 3:
                    rotateDir = -transform.up;
                    break;
                case 4:
                    rotateDir = transform.up;
                    break;
            }
        }
        if (shootCommand)
        {
            shoot = true;
            dirToGo *= 0.5f;

        }

        transform.position = dirToGo * moveSpeed;


        if (shoot)
        {
            Shot();
        }
    }
      

    // to be implemented by the develop
    public override void AgentReset()
    {
        academy.AcademyReset();

    }

    public void FixedUpdate()
    {
        WaitTimeInference();
    }

    private void WaitTimeInference()
    {
       

        if (!academy.GetIsInference())
        {
            RequestDecision();
        }
        else
        {
            if (timeSinceDecision >= timeBetweenDecisionsAtInference)
            {
                timeSinceDecision = 0f;

                if (brain == null)
                {
                    RuleBaseAction();
                    return;
                }

                RequestDecision();
            }
            else
            {
                timeSinceDecision += Time.fixedDeltaTime;
            }
        }
    }

    private void Shot()
    {
        var bullet = CombatAcademy.s_GameObjectPool.Alloc("resources/bullet", this,2f);

        bullet.transform.position = gameObject.transform.position;
        bullet.transform.rotation = gameObject.transform.rotation;
        bullet.GetComponent<Bullet>().SetOwnerAgent(this);
    }

    private void Turn(bool right)
    {
        if(right)
        {
            transform.Rotate(new Vector3(0, rotationAngle, 0));
        }
        else
        {
            transform.Rotate(new Vector3(0, -rotationAngle, 0));
        }
    }
}
