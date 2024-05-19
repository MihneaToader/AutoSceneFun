using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEngine;

public class RecordAlignment : MonoBehaviour
{
    public GameObject cameraToSave;
    public GameObject alignedObject;
    public GameObject meshAlign;
    private bool savedMesh = false;
    private JsonData data = new JsonData();

    [System.Serializable]
    public class TransformInfo
    {
        public Vector3 pos;
        public Quaternion rot;
        public Vector3 scale;

        public TransformInfo(Transform parentTransform)
        {
            pos = parentTransform.position;
            rot = parentTransform.rotation;
            scale = parentTransform.localScale;
        }
    }
    
    [System.Serializable]
    private class JsonData
    {
        public List<JsonEntry> entries;
    }

    [System.Serializable]
    private class JsonEntry
    {
        public string timestamp;
        public TransformInfo transformInfo;
    }
    
    void Start()
    {
        data.entries = new List<JsonEntry>();
        Debug.Log(Application.persistentDataPath);
    }

    void Update()
    {
        GameObject meshObj = GameObject.Find("ComputedMesh");
        if (meshAlign.GetComponent<MeshAlignment>().finishedAligning && savedMesh == false) 
        {
            SaveMeshToFile(meshObj);
            SaveMeshTransform(meshObj, "combinedMesh");
            SaveMeshTransform(alignedObject, "iPhoneMesh");
            savedMesh = true;
        }
        UpdateHeadset();
    }

    void OnApplicationQuit()
    {
        Debug.Log("Saving");
        SaveData();
    }

    void OnApplicationPause()
    {
        Debug.Log("Saving");
        SaveData();
    }

    void SaveMeshToFile(GameObject meshObj) 
    {
        string path = $"{Application.persistentDataPath}/roomMesh.obj";
        Mesh mesh = meshObj.GetComponent<MeshFilter>().sharedMesh;
        File.WriteAllText(path, GetMeshOBJ("roomMesh", mesh, meshObj.transform.localToWorldMatrix));
    }

    void SaveMeshTransform(GameObject meshObj, string name)
    {
        string path = $"{Application.persistentDataPath}/{name}.json";
        TransformInfo transformInfo = new TransformInfo(meshObj.transform);
        File.WriteAllText(path, JsonUtility.ToJson(transformInfo, true));
    }

    void UpdateHeadset()
    {
        string timestamp = System.DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss.fff");
        JsonEntry newEntry = new JsonEntry {timestamp = timestamp, transformInfo = new TransformInfo(cameraToSave.transform)};
        data.entries.Add(newEntry);
    }

    void SaveData()
    {
        string path = $"{Application.persistentDataPath}/headsetTracked.json";
        File.WriteAllText(path, JsonUtility.ToJson(data, true));
    }

    public string GetMeshOBJ(string name, Mesh mesh, Matrix4x4 objTransform)
    {
        StringBuilder sb = new StringBuilder();
 
        foreach (Vector3 v in mesh.vertices)
        {
            Vector3 writeV = objTransform.MultiplyPoint(v);
            sb.Append(string.Format("v {0} {1} {2}\n", writeV.x, writeV.y, writeV.z)); 
        }
        foreach (Vector3 v in mesh.normals)
            sb.Append(string.Format("vn {0} {1} {2}\n", v.x, v.y, v.z));
 
        for (int material = 0; material < mesh.subMeshCount; material++)
        {
            sb.Append(string.Format("\ng {0}\n", name));
            int[] triangles = mesh.GetTriangles(material);
            for (int i = 0; i < triangles.Length; i += 3)
            {
                sb.Append(string.Format("f {0}/{0} {1}/{1} {2}/{2}\n",
                triangles[i] + 1,
                triangles[i + 1] + 1,
                triangles[i + 2] + 1));
            }
        }
 
        return sb.ToString();
    }
}
