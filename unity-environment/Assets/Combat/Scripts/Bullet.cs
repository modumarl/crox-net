using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Bullet : MonoBehaviour
{

    public CombatAgent agent;
    public LineRenderer line;
    public float speed = 3;

    static GameObject resObjet;

    public GameObject owner;

    string targetTag = "";



    // Use this for initialization
    void Start()
    {

    }

    // Update is called once per frame
    void Update()
    {

    }

    public void SetOwnerAgent(CombatAgent agent_)
    {
        agent = agent_;

        if (agent_.tag == "agent")
        {
            targetTag = "enermy";

            SetColor(Color.blue);
        }
        else
        {
            targetTag = "agent";
            SetColor(Color.red);
        }
    }

    public void SetColor(Color color)
    {
        line.startColor = line.endColor = color;
    }

    public void OnDisable()
    {
        agent = null;
    }

    private void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject.tag == targetTag)
        {
            if (collision.gameObject.GetComponent<CombatAgent>().SetDamage())
                agent.AddReward(0.02f);
        }

        CombatAcademy.s_GameObjectPool.Release(gameObject);
    }

    private void FixedUpdate()
    {
        transform.position = transform.position + Time.fixedDeltaTime * speed * transform.forward;
    }

}