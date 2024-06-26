using System.Collections;
using System.Collections.Generic;
using System;
using System.Linq;
using UnityEngine;
using UnityEngine.UIElements;

public class PlaneDemo : MonoBehaviour
{

    public GameObject objToExtract;
    public const int maxIterationsRANSAC = 750;
    public const int maxIterationsFit = 100;
    public const float thresholdFit = 0.01f;
    public const float rotationStepSize = 1f;
    public const float positionStepSize = 0.2f;
    public const int numOfPlanes = 4;
    public const float distanceThreshold = 0.05f;
    public const float groundDelta = 0.01f;
    public const float VERT_THRESHOLD = 0.1f;

    private GameObject floor, wall1, wall2;
    private float floorHeight;

    // Start is called before the first frame update
    void Start()
    {
        if (objToExtract == null)
        {
            Debug.LogError("No mesh parent assigned");
            return;
        }

        OVRSceneManager sceneManager = FindObjectOfType<OVRSceneManager>();
        Mesh mesh = objToExtract.GetComponent<MeshFilter>().sharedMesh;
        // DisplayMesh(mesh, objToExtract.transform);
        List<Vector3> vertices = new List<Vector3>(mesh.vertices);
        List<Plane> planes = new List<Plane>();
        List<GameObject> planeObjects = new List<GameObject>();

        List<int> inliers = new List<int>();
        List<int> outliers = new List<int>();

        for (int j = 0; j < numOfPlanes; j++)
        {
            int maxNumInliers = 0;
            int minOutliers = Int32.MaxValue;
            Plane bestPlane = new Plane();
            List<int> maxInliers = new List<int>();
            
            for (int i = 0; i < maxIterationsRANSAC; i++)
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
                    if (Math.Abs(plane.GetDistanceToPoint(vertex)) < distanceThreshold)
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

            planeObjects.Add(DisplayPlane(bestPlane.flipped, objToExtract.transform));
            planes.Add(bestPlane);

            foreach (int index in maxInliers.OrderByDescending(v => v))
            {
                vertices.RemoveAt(index);
            }
        }
        LabelPlanes(planeObjects, planes);
    }

    void Update()
    {
        FitScene();
    }

    void LabelPlanes(List<GameObject> planeObjects, List<Plane> planes)
    {
        floor = planeObjects[0];
        wall1 = planeObjects[0];
        wall2 = planeObjects[0];
        Debug.Log("Now labeling " + planes.Count + " planes.");
        // Find the lowest plane to count for floor and furthest plane to count for wall.
        for (int i = 0; i < planes.Count; i++) {
            GameObject crtPlaneObj = planeObjects[i];
            Plane crtPlane = planes[i];
            Vector3 normal = crtPlane.normal;
            Debug.Log("Normal: " + normal);
            float distFromVertical = 1 - Math.Abs(normal.y);
            Debug.Log("Is vertical " + (distFromVertical < VERT_THRESHOLD));
            if (crtPlaneObj.transform.position.y < floor.transform.position.y && distFromVertical < VERT_THRESHOLD)
            {
                floor = crtPlaneObj;
            }
            if (Math.Abs(crtPlaneObj.transform.position.x) > Math.Abs(wall1.transform.position.x) && distFromVertical > VERT_THRESHOLD)
            {
                wall1 = crtPlaneObj;
            }
        }
        for (int i = 0; i < planes.Count; i++) {
            GameObject crtPlaneObj = planeObjects[i];
            Plane crtPlane = planes[i];
            Vector3 normal = crtPlane.normal;
            float distFromVertical = 1 - Math.Abs(normal.y);
            if (crtPlaneObj != wall1 && Math.Abs(crtPlaneObj.transform.position.z) > Math.Abs(wall2.transform.position.z) && distFromVertical > VERT_THRESHOLD)
            {
                wall2 = crtPlaneObj;
            }
        }
        floor.transform.rotation = Quaternion.FromToRotation(Vector3.up, Vector3.up) * objToExtract.transform.rotation;
        wall1.transform.parent = floor.transform;
        wall2.transform.parent = floor.transform;
        objToExtract.transform.parent = floor.transform;
        Renderer floorRenderer = floor.GetComponent<Renderer>();
        floorRenderer.material.color = new Color(1f, 0f, 0f, 0.75f);
        Renderer wall1Renderer = wall1.GetComponent<Renderer>();
        wall1Renderer.material.color = new Color(0f, 1f, 0f, 0.75f);
        Renderer wall2Renderer = wall2.GetComponent<Renderer>();
        wall2Renderer.material.color = new Color(0f, 0f, 1f, 0.75f);
    }

    void FitScene()
    {
        Vector3 newPosition = floor.transform.position;
        Vector3 newRotation = floor.transform.rotation.eulerAngles;
        Vector3 prevPosition = floor.transform.position;
        Vector3 prevRotation = floor.transform.rotation.eulerAngles;
        OVRSceneAnchor[] sceneAnchors = FindObjectsOfType<OVRSceneAnchor>();
        List<GameObject> walls = new List<GameObject>();
        if (sceneAnchors != null) 
        {
            for (int i = 0; i < sceneAnchors.Length; i++)
            {
                OVRSceneAnchor instance = sceneAnchors[i];
                OVRSemanticClassification classification = instance.GetComponent<OVRSemanticClassification>();
                if (classification != null)
                {
                    if (classification.Contains(OVRSceneManager.Classification.Floor))
                    {
                        floorHeight = instance.transform.position.y - groundDelta;
                        newPosition.y = floorHeight;
                    }
                    if (classification.Contains(OVRSceneManager.Classification.WallFace))
                    {
                        GameObject mesh = instance.gameObject;
                        walls.Add(mesh);
                    }
                }
            }
            // Evaluate metric
            float prevMetric = ComputeMetric(walls, prevPosition, prevRotation);
            Debug.Log("Prev metric: " + prevMetric);

            // Update parameters using gradient descent
            float metricForward = ComputeMetric(walls, prevPosition, prevRotation + rotationStepSize * Vector3.up);
            float metricBackward = ComputeMetric(walls, prevPosition, prevRotation - rotationStepSize * Vector3.up);
            float gradient = (metricForward - metricBackward) / (2 * rotationStepSize);
            newRotation = prevRotation - (gradient * rotationStepSize) * Vector3.up;

            Vector3[] directions = { Vector3.forward, Vector3.right };
            foreach (Vector3 direction in directions)
            {
                float metricForwardPos = ComputeMetric(walls, prevPosition + positionStepSize * direction, newRotation);
                float metricBackwardPos = ComputeMetric(walls, prevPosition - positionStepSize * direction, newRotation);
                float gradientPos = (metricForwardPos - metricBackwardPos) / (2 * positionStepSize);
                newPosition = prevPosition - (gradientPos * positionStepSize) * direction;
            }

            float metric = ComputeMetric(walls, newPosition, newRotation);
            if (metric < prevMetric)
            {
                floor.transform.position = newPosition;
                floor.transform.rotation = Quaternion.Euler(newRotation);
            } else {
                floor.transform.position = prevPosition;
                floor.transform.rotation = Quaternion.Euler(prevRotation);
            }
        }
    }

    float ComputeMetric(List<GameObject> walls, Vector3 position, Vector3 rotation)
    {   
        floor.transform.position = position;
        floor.transform.rotation = Quaternion.Euler(rotation);
        float minWall1Dist = float.MaxValue, minWall2Dist = float.MaxValue;
        foreach (GameObject wall in walls)
            {
                float wall1Distance = MeshDistanceFromPlane(wall1, wall);
                float wall2Distance = MeshDistanceFromPlane(wall2, wall);
                if (wall1Distance < minWall1Dist)
                {
                    minWall1Dist = wall1Distance;
                }
                if (wall2Distance < minWall2Dist)
                {
                    minWall2Dist = wall2Distance;
                }
            }

        return minWall1Dist + minWall2Dist;
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

    Vector3 ProjectPointOnPlane(Vector3 point, Plane plane)
    {
        float distance = plane.GetDistanceToPoint(point);
        return point + plane.normal * distance;
    }

    GameObject DisplayPlane(Plane plane, Transform parentTransform)
    {
        GameObject planeObject = GameObject.CreatePrimitive(PrimitiveType.Plane);
        planeObject.transform.position = plane.ClosestPointOnPlane(Vector3.zero) + parentTransform.position;
        planeObject.transform.rotation = Quaternion.FromToRotation(Vector3.up, plane.normal) * parentTransform.rotation;
        planeObject.transform.localScale = new Vector3(1.0f, 1.0f, 1.0f);

        // Apply transparent material
        Renderer renderer = planeObject.GetComponent<Renderer>();
        Material material = new Material(Shader.Find("Standard"));
        material.color = new Color(1f, 1f, 1f, 0.75f); // Adjust alpha for transparency
        material.doubleSidedGI = true; // Enable double-sided rendering
        renderer.material = material;

        return planeObject;
    }

    void DisplayMesh(Mesh mesh, Transform parentTransform)
    {
        GameObject meshObj = new GameObject("Mesh");
        //Add Components
        MeshFilter filter = meshObj.AddComponent<MeshFilter>();
        MeshRenderer renderer = meshObj.AddComponent<MeshRenderer>();
        meshObj.transform.position = parentTransform.position;
        meshObj.transform.rotation = parentTransform.rotation;
        meshObj.transform.localScale = parentTransform.localScale;
        filter.mesh = mesh;

        // Apply transparent material
        Material material = new Material(Shader.Find("Standard"));
        material.color = new Color(0f, 0f, 1f, 0.75f); // Adjust alpha for transparency
        material.doubleSidedGI = true; // Enable double-sided rendering
        renderer.material = material;
    }
}
