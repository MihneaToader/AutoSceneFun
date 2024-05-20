using UnityEngine;
using System.IO;
using System.Text;
using Newtonsoft.Json;
using System.Collections.Generic;
public class camera_position_recorder : MonoBehaviour
{
    private string filePath;
    private JsonData data;
    private float nextUpdateTime = 0f;
    private float updateInterval = 0.1f; // Update interval in seconds

    private void Start()
    {
        filePath = $"{Application.persistentDataPath}/{System.DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss")}_camera_position.json";
        Debug.Log($"File path: {filePath}");
        data = new JsonData();
        LoadData();
    }

    private void Update()
    {
        if (Time.time >= nextUpdateTime)
        {
            UpdateData();
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
            data.Entries = new System.Collections.Generic.List<JsonEntry>();
        }
    }

    private void UpdateData()
    {
        string timestamp = System.DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss:fff");
        Dictionary<string, Dictionary<string, float>> cameradict = new Dictionary<string, Dictionary<string, float>>();
        Dictionary<string, float> cameraData = new Dictionary<string, float>();
        

        // Record camera position
        cameraData["PositionX"] = Camera.main.transform.position.x;
        cameraData["PositionY"] = -Camera.main.transform.position.y;
        cameraData["PositionZ"] = Camera.main.transform.position.z;

        // Record camera rotation
        cameraData["RotationX"] = Camera.main.transform.rotation.x;
        cameraData["RotationY"] = Camera.main.transform.rotation.y;
        cameraData["RotationZ"] = Camera.main.transform.rotation.z;
        cameraData["RotationW"] = Camera.main.transform.rotation.w;
        cameradict["head"] = cameraData;
        
        data.Entries.Add(new JsonEntry { Timestamp = timestamp, Position_rotation = cameradict });
    }


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
        // public Dictionary<string, float> Position_rotation { get; set; }
        public Dictionary<string, Dictionary<string, float>> Position_rotation { get; set; }
    }
}