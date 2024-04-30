using UnityEngine;
using System.IO;
using System.Text;
using Newtonsoft.Json;
using System.Collections.Generic;

public class LogBoneLocation : MonoBehaviour
{
    private string filePath;
    private JsonData data;
    private float nextUpdateTime = 0f;
    private float updateInterval = 1f; // Update interval in seconds
    private List<GameObject> bonePoints = new List<GameObject>();
    [SerializeField] private float pointSize = 0.01f;

    [SerializeField] private OVRHand hand;
    [SerializeField] private OVRSkeleton handSkeleton;
    private void Awake()
    {
        if (!hand) hand = GetComponent<OVRHand>();
        if (!handSkeleton) handSkeleton = GetComponent<OVRSkeleton>();
    }
    private void Start()
    {
        filePath = $"{Application.persistentDataPath}/{System.DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss")}_hand_data.json";
        data = new JsonData();
        LoadData();
        CreateBonePoints();
    }

    
    
    private void Update()
    {
        hand = GetComponent<OVRHand>();
        if (Time.time >= nextUpdateTime)
        {
            UpdateData();
            SaveData();
            nextUpdateTime = Time.time + updateInterval;
            UpdateBonePoints();
        }
    }

    private void LoadData()
    {
        if (File.Exists(filePath))
        {
            string json = File.ReadAllText(filePath);
            data = JsonConvert.DeserializeObject<JsonData>(json);
        }
        else
        {
            data.Entries = new System.Collections.Generic.List<JsonEntry>();
        }
    }

    private void UpdateData()
    {
        string timestamp = System.DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss");
        Dictionary<string, Dictionary<string, float>> numbersPerBone = new Dictionary<string, Dictionary<string, float>>();
        foreach (var bone in handSkeleton.Bones)
        {
            Dictionary<string, float> boneData = new Dictionary<string, float>();
            boneData["PositionX"] = bone.Transform.position.x;
            boneData["PositionY"] = bone.Transform.position.y;
            boneData["PositionZ"] = bone.Transform.position.z;
            boneData["RotationX"] = bone.Transform.rotation.x;
            boneData["RotationY"] = bone.Transform.rotation.y;
            boneData["RotationZ"] = bone.Transform.rotation.z;
            boneData["RotationW"] = bone.Transform.rotation.w;
            numbersPerBone[$"{bone.Id}"] = boneData;
            
        }
        
        

        data.Entries.Add(new JsonEntry { Timestamp = timestamp, Position_rotation = numbersPerBone });
    }

    private void SaveData()
    {
        string json = JsonConvert.SerializeObject(data, Formatting.Indented);
        File.WriteAllText(filePath, json);
    }

    private class JsonData
    {
        public System.Collections.Generic.List<JsonEntry> Entries { get; set; }
    }

    private class JsonEntry
    {
        public string Timestamp { get; set; }
        public Dictionary<string, Dictionary<string, float>> Position_rotation { get; set; }
    }

    private void CreateBonePoints()
    {
        foreach (var bone in handSkeleton.Bones)
        {
            GameObject bonePoint = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            bonePoint.transform.localScale = Vector3.one * pointSize;
            bonePoint.name = $"Bone Point ({bone.Id})";
            bonePoints.Add(bonePoint);
        }
    }

    private void UpdateBonePoints()
    {
        string boneIds = "";
        for (int i = 0; i < handSkeleton.Bones.Count; i++)
        {
            var bone = handSkeleton.Bones[i];
            boneIds += $"{bone.Id}, ";
            var bonePoint = bonePoints[i];
            bonePoint.transform.position = bone.Transform.position;
        }

        // Remove the trailing comma and space
        if (boneIds.Length > 2)
            boneIds = boneIds.Substring(0, boneIds.Length - 2);

        // Debug.Log($"Bone IDs: {boneIds}");
    }
}