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
    private float updateInterval = 0.1f; // Update interval in seconds
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
        OVRSkeleton.SkeletonType handType = handSkeleton.GetSkeletonType();
        filePath = $"{Application.persistentDataPath}/{System.DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss")}_{handType}.json";
        Debug.Log($"File path: {filePath}");
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
        string timestamp = System.DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss:fff");
        Dictionary<string, Dictionary<string, float>> numbersPerBone = new Dictionary<string, Dictionary<string, float>>();
        foreach (var bone in handSkeleton.Bones)
        {
            Dictionary<string, float> boneData = new Dictionary<string, float>();
            boneData["PositionX"] = bone.Transform.position.x;
            boneData["PositionY"] = -bone.Transform.position.y;
            boneData["PositionZ"] = bone.Transform.position.z;
            boneData["RotationX"] = bone.Transform.rotation.x;
            boneData["RotationY"] = bone.Transform.rotation.y;
            boneData["RotationZ"] = bone.Transform.rotation.z;
            boneData["RotationW"] = bone.Transform.rotation.w;
            numbersPerBone[$"{bone.Id}"] = boneData;
            
        }
        
        

        data.Entries.Add(new JsonEntry { Timestamp = timestamp, Position_rotation = numbersPerBone });
    }
    // private void UpdateData()
    // {
    //     string timestamp = System.DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss:fff");
    //     Dictionary<string, Dictionary<string, float>> numbersPerBone = new Dictionary<string, Dictionary<string, float>>();

    //     // Get the camera's transform (assuming the camera is tagged as "MainCamera")
    //     Transform cameraTransform = Camera.main.transform;

    //     foreach (var bone in handSkeleton.Bones)
    //     {
    //         Dictionary<string, float> boneData = new Dictionary<string, float>();

    //         // Transform the position from camera space to world space
    //         Vector3 worldPosition = cameraTransform.TransformPoint(bone.Transform.position);
    //         boneData["PositionX"] = worldPosition.x;
    //         boneData["PositionY"] = worldPosition.y;
    //         boneData["PositionZ"] = worldPosition.z;

    //         // Transform the rotation from camera space to world space
    //         Quaternion worldRotation = cameraTransform.rotation * bone.Transform.rotation;
    //         boneData["RotationX"] = worldRotation.x;
    //         boneData["RotationY"] = worldRotation.y;
    //         boneData["RotationZ"] = worldRotation.z;
    //         boneData["RotationW"] = worldRotation.w;

    //         numbersPerBone[$"{bone.Id}"] = boneData;
    //     }

    //     data.Entries.Add(new JsonEntry { Timestamp = timestamp, Position_rotation = numbersPerBone });
    // }

    private void OnApplicationPause()
    {
        SaveData();
        
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