using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;


public class CombatAcademy : Academy
{
    [HideInInspector]
    public List<GameObject> listEnermy;
    [HideInInspector]
    public int[] players;


    //[HideInInspector]
    public GameObject[] trueAgent;

    public int gridSize;

    public GameObject camObject;
    Camera cam;
    Camera agentCam;

    public GameObject agentPref;
    public GameObject enermyPref;

    public GameObject[] prefebs;

    public UnityEngine.UI.Text debugText;

    

    GameObject plane;
    GameObject sN;
    GameObject sS;
    GameObject sE;
    GameObject sW;

    CsvFileWriter csvWriter;

    public static GameObjPool s_GameObjectPool = null;

    int nGameNo =1;

    public override void InitializeAcademy()
    {
        if(s_GameObjectPool ==null)
        {
            s_GameObjectPool = new GameObjPool();

            if(prefebs!=null&& prefebs.Length>0)
            {
                foreach( var item in prefebs)
                {
                    s_GameObjectPool.LoadResource(item, 0);
                }
            }
        }


        gridSize = (int)resetParameters["gridSize"];
        cam = camObject.GetComponent<Camera>();

      

        agentCam = GameObject.Find("agentCam").GetComponent<Camera>();

        listEnermy = new List<GameObject>();

        plane = GameObject.Find("Plane");
        sN = GameObject.Find("sN");
        sS = GameObject.Find("sS");
        sW = GameObject.Find("sW");
        sE = GameObject.Find("sE");


        int AgenCount = (int)resetParameters["agent"];

        trueAgent = new GameObject[AgenCount];



        var brain = transform.Find("CombatBrain").GetComponent<Brain>();
       
        for (int i = 0; i < AgenCount; ++i)
        {
            trueAgent[i] = Instantiate(agentPref);
            trueAgent[i].GetComponent<CombatAgent>().agentParameters.agentCameras[0] =
                GameObject.Find("agentCam").GetComponent<Camera>();

            trueAgent[i].GetComponent<CombatAgent>().brain = brain;

        }

       
        for (int i = 0; i < (int)resetParameters["enermy"]; i++)
        {
            listEnermy.Add(Instantiate(enermyPref));       
        }

        SetEnvironment();


    }

    private void Start()
    {


        var brain = transform.Find("CombatBrain").GetComponent<Brain>();

      
      


    }

    public void SetEnvironment()
    {
        cam.transform.position = new Vector3(-((int)resetParameters["gridSize"] - 1) / 2f, 
                                             (int)resetParameters["gridSize"] * 1.25f, 
                                             -((int)resetParameters["gridSize"] - 1) / 2f);
        cam.orthographicSize = ((int)resetParameters["gridSize"] + 5f) / 2f;

       


    
        plane.transform.localScale = new Vector3(gridSize / 10.0f, 1f, gridSize / 10.0f);
        plane.transform.position = new Vector3((gridSize - 1) / 2f, -0.5f, (gridSize - 1) / 2f);
        sN.transform.localScale = new Vector3(1, 1, gridSize + 2);
        sS.transform.localScale = new Vector3(1, 1, gridSize + 2);
        sN.transform.position = new Vector3((gridSize - 1) / 2f, 0.0f, gridSize);
        sS.transform.position = new Vector3((gridSize - 1) / 2f, 0.0f, -1);
        sE.transform.localScale = new Vector3(1, 1, gridSize + 2);
        sW.transform.localScale = new Vector3(1, 1, gridSize + 2);
        sE.transform.position = new Vector3(gridSize, 0.0f, (gridSize - 1) / 2f);
        sW.transform.position = new Vector3(-1, 0.0f, (gridSize - 1) / 2f);

        agentCam.orthographicSize = (gridSize) / 2f;
        agentCam.transform.position = new Vector3((gridSize - 1) / 2f, gridSize + 1f, (gridSize - 1) / 2f);

    }

    public override void AcademyReset()
    {
      
        HashSet<Vector3> hashPos = new HashSet<Vector3>();

        System.Func<GameObject ,bool> UniqPos = (GameObject obj) =>
        {

            while(true)
            {
                int r = Random.Range(0, gridSize * gridSize);
                int x = (r) / gridSize;
                int y = (r) % gridSize;


                Vector3 targetPos = new Vector3(x, -0.25f, y);


                if (!hashPos.Contains(targetPos))
                {
                    obj.transform.position = targetPos;
                    hashPos.Add(targetPos);
                    return true;
                }
            }

            return false;
      
        };



        for (int i = 0; i < trueAgent.Length; ++i)
        {
            if (trueAgent[i] == null)
                continue;

            UniqPos(trueAgent[i]);

            trueAgent[i].GetComponent<CombatAgent>().Init();
      

        }


        foreach(var o in listEnermy)
        {
            UniqPos(o);
        }

    
    }

    public override void AcademyStep()
    {


    }

    public void WriteSummary(int step)
    {

        return;

        if (csvWriter == null)
            return;

        {
            List<string> columns = new List<string>();

            columns.Add(nGameNo.ToString());

            float totalReward = 0;
            for (int i = 0; i < trueAgent.Length; ++i)
            {

                totalReward += trueAgent[i].GetComponent<GridAgent>().GetReward();

            }


            columns.Add(totalReward.ToString());

            columns.Add(step.ToString());

            csvWriter.WriteRow(columns);
            nGameNo++;
        }
    }

    public void SetDone()
    {
     

    }


    public void UIDeBugText(string str)
    {
        debugText.text = str;
    }
}
