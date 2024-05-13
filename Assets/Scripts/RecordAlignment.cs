using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEngine;
using Newtonsoft.Json;

public class RecordAlignment : MonoBehaviour
{
    public GameObject cameraToSave;
    public GameObject alignedObject;
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

    private class JsonData
    {
        public List<JsonEntry> entries { get; set; }
    }

    private class JsonEntry
    {
        public string timestamp { get; set; }
        public TransformInfo transformInfo {get; set; }
    }
    
    void Update()
    {
        GameObject meshObj = GameObject.Find("ComputedMesh");
        if (meshObj != null && savedMesh == false) 
        {
            Mesh meshToSave = meshObj.GetComponent<MeshFilter>().sharedMesh;
            SaveMeshToFile(meshToSave);
            SaveMeshTransform(meshObj, "combinedMesh");
            SaveMeshTransform(alignedObject, "iPhoneMesh");
            savedMesh = true;
        }
        UpdateHeadset();
    }

    void Destroy()
    {
        SaveData();
    }

    void SaveMeshToFile(Mesh mesh) 
    {
        string path = $"{Application.persistentDataPath}/roomMesh.obj";
        File.WriteAllText(path, GetMeshOBJ("roomMesh", mesh));
    }

    void SaveMeshTransform(GameObject meshObj, string name)
    {
        string path = $"{Application.persistentDataPath}/{name}.txt";
        TransformInfo transformInfo = new TransformInfo(meshObj.transform);
        File.WriteAllText(path, JsonUtility.ToJson(transformInfo, true));
    }

    void UpdateHeadset()
    {
        string timestamp = System.DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss");
        data.entries.Add(new JsonEntry {timestamp = timestamp, transformInfo = new TransformInfo(cameraToSave.transform)});
    }

    void SaveData()
    {
        string path = $"{Application.persistentDataPath}/headsetTracked.json";
        string json = JsonConvert.SerializeObject(data, Formatting.Indented);
        File.WriteAllText(path, json);
    }

    public string GetMeshOBJ(string name, Mesh mesh)
    {
        StringBuilder sb = new StringBuilder();
 
        foreach (Vector3 v in mesh.vertices)
            sb.Append(string.Format("v {0} {1} {2}\n", v.x, v.y, v.z));
 
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
