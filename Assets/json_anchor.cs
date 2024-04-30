using UnityEngine;
using System.IO;
using System.Text;
using Newtonsoft.Json;
using System.Collections.Generic;

public class json_anchor : MonoBehaviour
{
    private string filePath;
    private JsonData data;
    private float nextUpdateTime = 0f;
    private float updateInterval = 1f; // Update interval in seconds

    private void Awake()
    {
        // No need for OVRHand or OVRSkeleton here
    }

    private void Start()
    {
        filePath = Application.persistentDataPath + "/scene_anchors.json";
        data = new JsonData();
        LoadData();
    }

    private void Update()
    {
        if (Time.time >= nextUpdateTime)
        {
            UpdateData();
            SaveData();
            nextUpdateTime = Time.time + updateInterval;
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
            data.Entries = new List<JsonEntry>();
        }
    }

    private void UpdateData()
    {
        string timestamp = System.DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss");
        Dictionary<string, Dictionary<string, float>> anchorsData = new Dictionary<string, Dictionary<string, float>>();

        OVRSceneAnchor[] sceneAnchors = FindObjectsOfType<OVRSceneAnchor>();

        for (int i = 0; i < sceneAnchors.Length; i++)
        {
            OVRSceneAnchor anchor = sceneAnchors[i];
            Dictionary<string, float> anchorData = new Dictionary<string, float>();

            anchorData["PositionX"] = anchor.transform.position.x;
            anchorData["PositionY"] = anchor.transform.position.y;
            anchorData["PositionZ"] = anchor.transform.position.z;
            anchorData["RotationX"] = anchor.transform.rotation.x;
            anchorData["RotationY"] = anchor.transform.rotation.y;
            anchorData["RotationZ"] = anchor.transform.rotation.z;
            anchorData["RotationW"] = anchor.transform.rotation.w;

            anchorsData[$"Anchor{i}"] = anchorData;
        }

        data.Entries.Add(new JsonEntry { Timestamp = timestamp, Position_rotation = anchorsData });
    }

    private void SaveData()
    {
        string json = JsonConvert.SerializeObject(data, Formatting.Indented);
        File.WriteAllText(filePath, json);
    }

    private class JsonData
    {
        public List<JsonEntry> Entries { get; set; }
    }

    private class JsonEntry
    {
        public string Timestamp { get; set; }
        public Dictionary<string, Dictionary<string, float>> Position_rotation { get; set; }
    }
}