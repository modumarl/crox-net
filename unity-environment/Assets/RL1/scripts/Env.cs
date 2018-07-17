using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Common;

public class Env :  Singleton<Env>
{

    // Use this for initialization

    public BoxCollider chaserStart;
    public BoxCollider targetStart;

    public GameObject chaserRes;
    public GameObject targetRes;

    public int chaserCount = 2;
    public int targetCount = 1;

    public List<GameObject>[] listActor = new List<GameObject>[(int)Common.ACTOR_TYPE.End];

    public static GameObjPool s_ObjPool;


    bool bDone = false;

    public bool IsDone
    {
        get
        {
            return bDone;
        }
    }


    void Start ()
    {
        if (s_ObjPool == null)
        {
            s_ObjPool = new GameObjPool();
            s_ObjPool.LoadResource(chaserRes, 2);
            s_ObjPool.LoadResource(targetRes, 1);
        }

        for (int i = 0; i < listActor.Length; ++i)
        {
            listActor[i] = new List<GameObject>();
        }



        Reset();
    }

    private void Awake()
    {
        
    }

    // Update is called once per frame
    void Update ()
    {
		
	}

    public void ReleaseActor()
    {
        for(int i=0;i< listActor.Length;++i)
        {
            foreach (var o in listActor[i])
            {
                s_ObjPool.Release(o);
            }

            listActor[i].Clear();
        }   
    }

    public void Reset()
    {
        ReleaseActor();

        SummonActor(chaserRes, chaserCount, chaserStart);
        SummonActor(targetRes, targetCount, targetStart);
    }

    public void SummonActor(GameObject res,int count, BoxCollider area)
    {
        area.enabled = true;

        for (int i = 0; i < count; ++i)
        {
            GameObject actor = s_ObjPool.Alloc(res.name, this);

            Vector3 rndPos = new Vector3(Random.Range(area.bounds.min.x, area.bounds.max.x),
                                      area.bounds.center.y,
                                      Random.Range(area.bounds.min.z, area.bounds.max.z));

            actor.transform.position = rndPos;
            actor.transform.Rotate(0,Random.Range(0, 360),0);

            if (actor)
            {
                listActor[(int)actor.GetComponent<ActorObj>().actorType].Add(actor);
            }
        }

        area.enabled = false;
    }



}
