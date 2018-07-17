
//#define __USE_NAVMESH

using System.Collections.Generic;
using UnityEngine;
using UnityEngine.AI;
using Common;



public class ActorObj : MonoBehaviour
{

    // Use this for initialization

    public Common.ACTOR_TYPE actorType;
    public Common.CTRL_TYPE ctrlType;

    //public float speed = 3;              //초당 이동속도
    public float processInteval = 0.1f;  //0.1초

    public static int dirCount =  16;

    public NavMeshAgent navMeshAgent;

    static Vector3[] dir;
    static float dirStep;


    float processTime = 0;
    bool bVoidObstacle = false;
    int nCurDir =-1;
 
    GameObject enermy = null;

    static Vector3 DegreeToVector(float degree)
    {
        float radian = degree * Mathf.Deg2Rad;
        Vector3 direction = new Vector3(Mathf.Sin(radian), 0, Mathf.Cos(radian));

        return direction;
    }

    void Start ()
    {
        
        if(dir == null)
        {
            dirStep = 360f / dirCount;

            dir = new Vector3[dirCount];
      
            for (int i=0;i<dir.Length;++i)
            {
                dir[i] = DegreeToVector(dirStep * i);
            }
        }

        if (navMeshAgent == null)
            navMeshAgent = GetComponent<NavMeshAgent>();


    }

    private void OnEnable()
    {
        processTime = 0;
        enermy = null;
    }

    // Update is called once per frame
    void Update ()
    {

#if __USE_NAVMESH
        if (enermy&&enermy.activeSelf&& navMeshAgent.destination != enermy.transform.position)
            navMeshAgent.SetDestination(enermy.transform.position);
#endif

        if (Time.time - processTime < processInteval)
            return;


        processTime = Time.time;

        switch(actorType)
        {
            case ACTOR_TYPE.chaser:
                ProcChaser();
                break;

            case ACTOR_TYPE.target:
             //   ProcTarget();
                break;

        }

    }

    void FindEermy(List<GameObject> list,out GameObject enermy, out float xDist)
    {
        float minXDist = 10000f;

        enermy = null;

        foreach (var o in list)
        {
            if (o == null || o.activeSelf == false)
                continue;

            float dist = Vector3.SqrMagnitude(gameObject.transform.position - o.transform.position);

            if (dist < minXDist)
            {
                minXDist = dist;
                enermy = o;
            }

        }

        xDist = minXDist;

    }


    float GetFromDir(GameObject a, GameObject b)
    {

        Quaternion q = Quaternion.LookRotation(new Vector3(
                a.transform.position.x - b.transform.position.x,
                0, a.transform.position.z - b.transform.position.z));


        float angle = q.eulerAngles.y;
        if (angle < 0)
        {
            angle += 360;
        }

        return angle;

    }




    public void LookAt(Vector3 direction)
    {
        if (direction.x == 0 && direction.z == 0) return;

        Quaternion lookAt = Quaternion.LookRotation(new Vector3(direction.x, 0, direction.z));

        transform.rotation = lookAt;
    }

    public void LookAt(float degree)
    {
        LookAt(DegreeToVector(degree));
    }



    bool UpdatePos()
    {

#if __USE_NAVMESH
        return false;

#else


        Vector3 nextPos = transform.position + transform.forward * navMeshAgent.speed * processInteval;

        nextPos.y = nextPos.y + 2;

        Ray ray = new Ray(nextPos,Vector3.down);
        RaycastHit hit;

      
        if (Physics.Raycast(ray.origin, ray.direction, out hit, 100f, 1 << LayerMask.NameToLayer("Ground")) == false)
            return false;


         transform.position = hit.point;

        return true;
#endif
    }

    void ProcChaser()
    {
        var list = Env.Inst.listActor[(int)ACTOR_TYPE.target];

        if (list == null|| list.Count == 0)
            return;


        float xDist;

        FindEermy(list,out enermy,out xDist);

        if (enermy == null)
            return;


      

        float angle = GetFromDir(enermy, gameObject);

        int nDir = (int)(angle / dirStep);



        Ray ray = new Ray(transform.position, dir[nDir]);
        RaycastHit hit;
        if (Physics.Raycast(ray.origin, ray.direction, out hit, 100f, 1 << LayerMask.NameToLayer("Obstacle")) == true)
        {

            if (nCurDir < 0)
            {
                nCurDir = nDir;
            }

            LookAt(dir[nCurDir]);
            nCurDir = FindOpenPos(nCurDir);
            return;
        }

        LookAt(dir[nDir]);

        FindOpenPos(nDir);

        //UpdatePos();

        // transform.position = transform.position + transform.forward * speed * processInteval;

    }

    void ProcTarget()
    {
        var list = Env.Inst.listActor[(int)ACTOR_TYPE.chaser];

        if (list == null || list.Count == 0)
            return;

        float xDist;

        FindEermy(list, out enermy, out xDist);

        if (enermy == null)
            return;


        if (xDist > 40)
            return;


        float angle = GetFromDir(gameObject,enermy);

        int nDir = (int)(angle / dirStep);
       // LookAt(dir[nDir]);

       // UpdatePos();

       // transform.position = transform.position + transform.forward * speed * processInteval;

    }

    bool IsMoveable(Vector3 dir,out Vector3 nextPos)
    {
        Ray ray = new Ray(transform.position, dir);
        RaycastHit hit;
        nextPos = transform.position;
        if (Physics.Raycast(ray.origin, ray.direction, out hit, 100f, 1 << LayerMask.NameToLayer("Obstacle")) == true)
        {      
            return false;
        }

        Vector3 temp = transform.position + dir * navMeshAgent.speed * processInteval;
        temp.y = temp.y + 2;

        ray.origin = temp; ray.direction = Vector3.down;

        if (Physics.Raycast(ray.origin, ray.direction, out hit, 100f, 1 << LayerMask.NameToLayer("Ground")) == false)
            return false;

        nextPos = hit.point;
        return true;
    }


    int FindOpenPos(int nDir)
    {
        int nRand = Random.Range(0, 1);

        int[] wise = new int[2];


        ///지그제그로 Angle을 검색하기 위해서

        if(nRand>0)
        {
            wise[0] = -1; wise[1] = 1;
        }
        else
        {
            wise[0] = 1; wise[1] = -1;
        }


        int newDir = -1;


        for(int i = 0;i< dirCount;++i)
        {
            
            int rest = i % 2;
            int j = i >> 1;

            newDir = (j + rest)* wise[rest];

            newDir += nDir;

            if (newDir >= dirCount)
            {
                newDir -= dirCount;
            }
            else if(newDir < 0)
            {
                newDir += dirCount;
            }

            Vector3 nextPos = new Vector3();

            if (!IsMoveable(dir[newDir], out nextPos))
                continue;

            transform.position= nextPos;

            break;

        }


        if(newDir == nDir)
            bVoidObstacle = false;


        return newDir;


    }


 
}
