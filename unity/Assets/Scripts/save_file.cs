using System;
using System.IO;
using UnityEngine;

public class FileCreator : MonoBehaviour
{
    private string _path;

    void Start()
    {
        _path = Application.persistentDataPath + "/123.txt";

        if (!File.Exists(_path))
        {
            File.WriteAllText(_path, "My string text");
            Debug.Log("File created at: " + _path);
        }
        else
        {
            Debug.Log("File already exists at: " + _path);
        }
    }
}
