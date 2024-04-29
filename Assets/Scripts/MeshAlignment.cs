using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using g3;
using System.Collections.Specialized;
using System.Linq;

public class MeshAlignment : MonoBehaviour
{
    public GameObject sourceMeshObject;
    private MeshFilter sourceMeshFilter;
    private GameObject currentlyAlignedMesh = null;

    void Start()
    {
        sourceMeshFilter = sourceMeshObject.GetComponent<MeshFilter>();
    }

    void Update()
    {
        OVRSceneAnchor[] sceneAnchors = FindObjectsOfType<OVRSceneAnchor>();
        List<MeshFilter> targetMeshes = new List<MeshFilter>();
        if (sceneAnchors != null) 
        {
            for (int i = 0; i < sceneAnchors.Length; i++)
                {
                    OVRSceneAnchor instance = sceneAnchors[i];
                    OVRSemanticClassification classification = instance.GetComponent<OVRSemanticClassification>();
                    if (classification != null)
                    {
                        targetMeshes.Add(instance.gameObject.GetComponent<MeshFilter>());
                    }
                }
        }
        if (targetMeshes.Count != 0)
        {
            if (currentlyAlignedMesh != null)
            {
                Destroy(currentlyAlignedMesh);
            }
            Mesh targetMesh = CombineMeshes(targetMeshes);
            currentlyAlignedMesh = AlignMeshes(targetMesh);
        }
    }

    public Mesh CombineMeshes(List<MeshFilter> meshFilters)
    {
        CombineInstance[] combineInstances = new CombineInstance[meshFilters.Count];
        for (int i = 0; i < meshFilters.Count; i++)
        {
            combineInstances[i].mesh = meshFilters[i].sharedMesh;
            combineInstances[i].transform = meshFilters[i].transform.localToWorldMatrix;
            // Destroy(meshFilters[i].gameObject);
        }

        // Combine meshes into a single mesh
        Mesh combinedMesh = new Mesh();
        combinedMesh.CombineMeshes(combineInstances, true, true);

        return combinedMesh;
    }

    public GameObject AlignMeshes(Mesh targetMesh)
    {
        // Make sure both source and target mesh filters are not null
        if (sourceMeshFilter == null || targetMesh == null)
        {
            Debug.LogError("Mesh filters are not assigned!");
            return null;
        }

        // Get the mesh data
        Mesh sourceMesh = sourceMeshFilter.sharedMesh;

        // Convert Unity meshes to geometry3Sharp meshes
        DMesh3 sourceDMesh = UnityMeshToDMesh(sourceMesh);
        DMesh3 targetDMesh = UnityMeshToDMesh(targetMesh);
        DMeshAABBTree3 targetTree = new DMeshAABBTree3(targetDMesh);
        targetTree.Build();

        // Perform ICP alignment
        MeshICP icp = new MeshICP(sourceDMesh, targetTree);

        // Apply the transformation to the source mesh
        icp.Solve();

        icp.UpdateVertices(sourceDMesh);

        // Convert the aligned geometry3Sharp mesh back to Unity mesh
        Mesh newMesh = DMeshToUnityMesh(sourceDMesh);
        return DisplayMesh(newMesh, transform);
    }

    // Convert Unity.Mesh to g3.DMesh3
    private DMesh3 UnityMeshToDMesh(Mesh unityMesh)
    {
        var vertices = unityMesh.vertices;
        var triangles = unityMesh.triangles;
        var normals = unityMesh.normals;
        DMesh3 dMesh = DMesh3Builder.Build(vertices, triangles, normals);
        return dMesh;
    }

    // Convert g3.DMesh3 to Unity.Mesh
    private Mesh DMeshToUnityMesh(DMesh3 dMesh)
    {
        Mesh unityMesh = new Mesh();
        g3.Vector3d[] vertices = dMesh.Vertices().ToArray();
        int[] triangles = dMesh.TrianglesBuffer.ToArray();
        Vector3[] unityVertices = new Vector3[vertices.Length];
        for (int i = 0; i < vertices.Length; i++)
        {
            unityVertices[i] = new Vector3((float)vertices[i].x, (float)vertices[i].y, (float)vertices[i].z);
        }
        unityMesh.vertices = unityVertices;
        unityMesh.triangles = triangles;
        unityMesh.RecalculateNormals();
        return unityMesh;
    }

    GameObject DisplayMesh(Mesh mesh, Transform parentTransform)
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

        return meshObj;
    }
}
