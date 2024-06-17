using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using System.Linq;
using Unity.VisualScripting;

public class PlaneDistance : MonoBehaviour
{
    public const float distanceThreshold = 0.1f;

    public GameObject meshObj1, meshObj2, planeObj;

    // Start is called before the first frame update
    void Start()
    {
        // if (objToExtract == null)
        // {
        //     Debug.LogError("No mesh parent assigned");
        //     return;
        // }

        // if (planeToCalculate == null)
        // {
        //     Debug.LogError("No plane object assigned");
        //     return;
        // }
        // Mesh mesh = objToExtract.GetComponent<MeshFilter>().sharedMesh;
        // List<Vector3> vertices = new List<Vector3>(mesh.vertices);

        // var filter = planeToCalculate.GetComponent<MeshFilter>();
        // var transform = planeToCalculate.GetComponent<Transform>();
        // Vector3 normal = filter.transform.TransformDirection(filter.mesh.normals[0]);
        // Plane plane = new Plane(normal, transform.position);

        // List<int> inliers = new List<int>();
        // List<int> outliers = new List<int>();
        // for (var k = 0; k < vertices.Count; k++)
        // {
        //     Vector3 vertex = vertices[k];
        //     if (Math.Abs(plane.GetDistanceToPoint(vertex)) < distanceThreshold)
        //     {
        //         inliers.Add(k);
        //     }
        //     else
        //     {
        //         outliers.Add(k);
        //     }
        // }

        // Debug.Log("Plane has " + inliers.Count + " inliers and " + outliers.Count + "outliers.");
        // Debug.Log("Plane distance: " + plane.distance + "; Plane normal: " + plane.normal);

        // DisplayPlane(plane);
        // DisplayMesh(mesh);
    }

    void Update() 
    {
        Debug.Log("Distance from wall 1: " + MeshDistanceFromPlane(planeObj, meshObj1));
        Debug.Log("Distance from wall 2: " + MeshDistanceFromPlane(planeObj, meshObj2));
    }

    void DisplayPlane(Plane plane)
    {
        GameObject planeObject = GameObject.CreatePrimitive(PrimitiveType.Plane);
        planeObject.transform.position = plane.ClosestPointOnPlane(Vector3.zero);
        planeObject.transform.rotation = Quaternion.FromToRotation(Vector3.up, plane.normal);

        // Apply transparent material
        Renderer renderer = planeObject.GetComponent<Renderer>();
        Material material = new Material(Shader.Find("Standard"));
        material.color = new Color(1f, 0f, 0f, 0.75f); // Adjust alpha for transparency
        material.doubleSidedGI = true; // Enable double-sided rendering
        renderer.material = material;
    }

    void DisplayMesh(Mesh mesh)
    {
        GameObject meshObj = new GameObject("Mesh");
        //Add Components
        MeshFilter filter = meshObj.AddComponent<MeshFilter>();
        MeshRenderer renderer = meshObj.AddComponent<MeshRenderer>();

        filter.mesh = mesh;

        // Apply transparent material
        Material material = new Material(Shader.Find("Standard"));
        material.color = new Color(0f, 0f, 1f, 0.75f); // Adjust alpha for transparency
        material.doubleSidedGI = true; // Enable double-sided rendering
        renderer.material = material;
    }

    float MeshDistanceFromPlane(GameObject meshObj, GameObject planeObj)
    {
        Mesh mesh = meshObj.GetComponent<MeshFilter>().sharedMesh;
        Transform meshTransform = meshObj.transform;
        MeshFilter filter = planeObj.GetComponent<MeshFilter>();
        Vector3 normal = Vector3.up;
        if(filter && filter.mesh.normals.Length > 0)
            normal = filter.transform.TransformDirection(filter.mesh.normals[0]);
        Plane plane = new Plane(normal, planeObj.transform.position);

        List<float> distances = new List<float>();
        for (int i = 0; i < mesh.vertices.Length; i++) {
            Vector3 vertex = meshTransform.TransformPoint(mesh.vertices[i]);
            float distance = plane.GetDistanceToPoint(vertex);
            distances.Add(Math.Abs(distance));
        }
        distances = distances.OrderByDescending(v => v).ToList();
        return distances[0] + distances[1] + distances[3];
    }
}
