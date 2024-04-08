using System.Collections;
using System.Collections.Generic;
using System;
using System.Linq;
using UnityEngine;

public class PlaneDemo : MonoBehaviour
{

    public GameObject objToExtract;
    public const int maxIterations = 1000;
    public const int numOfPlanes = 4;
    public const float distanceThreshold = 0.0000001f;

    // Start is called before the first frame update
    void Start()
    {
        if (objToExtract == null)
        {
            Debug.LogError("No mesh parent assigned");
            return;
        }

        Mesh mesh = objToExtract.GetComponent<MeshFilter>().sharedMesh;
        List<Vector3> vertices = new List<Vector3>(mesh.vertices);

        List<int> inliers = new List<int>();
        List<int> outliers = new List<int>();

        for (int j = 0; j < numOfPlanes; j++)
        {
            int maxNumInliers = 0;
            int minOutliers = Int32.MaxValue;
            Plane bestPlane = new Plane();
            List<int> maxInliers = new List<int>();
            
            for (int i = 0; i < maxIterations; i++)
            {
                // Randomly select three points
                int index1 = UnityEngine.Random.Range(0, vertices.Count);
                int index2 = UnityEngine.Random.Range(0, vertices.Count);
                int index3 = UnityEngine.Random.Range(0, vertices.Count);

                Vector3 p1 = vertices[index1];
                Vector3 p2 = vertices[index2];
                Vector3 p3 = vertices[index3];

                // Fit a plane using these three points
                Plane plane = new Plane(p1, p2, p3);

                // Count inliers and outliers
                for (var k = 0; k < vertices.Count; k++)
                {
                    Vector3 vertex = vertices[k];
                    if (plane.GetDistanceToPoint(vertex) < distanceThreshold)
                    {
                        inliers.Add(k);
                    }
                    else
                    {
                        outliers.Add(k);
                    }
                }

                if (inliers.Count > maxNumInliers)
                {
                    maxNumInliers = inliers.Count;
                    maxInliers = new List<int>(inliers);
                    minOutliers = outliers.Count;
                    bestPlane = plane;
                }

                inliers.Clear();
                outliers.Clear();
            }
            Debug.Log("Plane found with " + maxNumInliers + " inliers and " + minOutliers + "outliers.");
            Debug.Log("Plane distance: " + bestPlane.distance + "; Plane normal: " + bestPlane.normal);

            DisplayPlane(bestPlane.flipped);

            foreach (int index in maxInliers.OrderByDescending(v => v))
            {
                vertices.RemoveAt(index);
            }
        }
    }

    void DisplayPlane(Plane plane)
    {
        GameObject planeObject = GameObject.CreatePrimitive(PrimitiveType.Plane);
        planeObject.transform.position = plane.ClosestPointOnPlane(Vector3.zero);
        planeObject.transform.rotation = Quaternion.FromToRotation(Vector3.up, plane.normal);

        // Adjust the scale of the plane to fit the mesh size
        Bounds meshBounds = objToExtract.GetComponent<MeshRenderer>().bounds;
        float scaleX = meshBounds.size.x;
        float scaleZ = meshBounds.size.z;
        planeObject.transform.localScale = new Vector3(scaleX, 1f, scaleZ);

        // Apply transparent material
        Renderer renderer = planeObject.GetComponent<Renderer>();
        Material material = new Material(Shader.Find("Standard"));
        material.color = new Color(1f, 0f, 0f, 0.75f); // Adjust alpha for transparency
        material.doubleSidedGI = true; // Enable double-sided rendering
        renderer.material = material;
    }
}
