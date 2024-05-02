using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using g3;
using System.Collections.Specialized;
using System.Linq;
using System.Threading.Tasks;

public class MeshAlignment : MonoBehaviour
{
    public GameObject sourceMeshObject;
    private MeshFilter sourceMeshFilter;
    DMeshAABBTree3 targetTree;
    DMesh3 sourceDMesh;
    MeshICP icp;
    int frameWait = 0;
    bool sceneModelLoaded = false;
    bool savedTarget = false;
    bool isAligning = false;

    void Start()
    {
        sourceMeshFilter = sourceMeshObject.GetComponent<MeshFilter>();
        var sceneManager = FindObjectOfType<OVRSceneManager>();
        sceneManager.SceneModelLoadedSuccessfully += SceneModelLoaded;
    }

    void SceneModelLoaded()
    {
        sceneModelLoaded = true;
    }

    void Update()
    {
        if (sceneModelLoaded)
        {
            if (!savedTarget)
            {
                if (frameWait < 10)
                {
                    frameWait++;
                    return;
                }

                List<OVRSceneAnchor> sceneAnchors = new List<OVRSceneAnchor>();
                OVRSceneAnchor.GetSceneAnchors(sceneAnchors);
                List<MeshFilter> targetMeshes = new List<MeshFilter>();
                List<GameObject> toDestroy = new List<GameObject>();
                if (sceneAnchors != null) 
                {
                    for (int i = 0; i < sceneAnchors.Count; i++)
                        {
                            OVRSceneAnchor instance = sceneAnchors[i];
                            OVRSemanticClassification classification = instance.GetComponent<OVRSemanticClassification>();
                            if (classification != null)
                            {
                                MeshFilter meshFilter = instance.gameObject.GetComponent<MeshFilter>();
                                OVRSceneVolume volume = instance.gameObject.GetComponent<OVRSceneVolume>();
                                if (meshFilter != null && meshFilter.sharedMesh != null)
                                    targetMeshes.Add(meshFilter);
                                if (volume != null)
                                {
                                    GameObject volumeObject = VolumeToMesh(volume);
                                    targetMeshes.Add(volumeObject.GetComponent<MeshFilter>());
                                    toDestroy.Add(volumeObject);
                                }
                            }
                        }
                }
                if (targetMeshes.Count != 0)
                {
                    Debug.Log("Number of anchors: " + targetMeshes.Count);
                    Mesh targetMesh = CombineMeshes(targetMeshes);
                    foreach (GameObject crtObj in toDestroy)
                        Destroy(crtObj);
                    DMesh3 targetDMesh = UnityMeshToDMesh(targetMesh);
                    sourceDMesh = UnityMeshToDMesh(sourceMeshFilter.sharedMesh);
                    targetTree = new DMeshAABBTree3(targetDMesh);
                    targetTree.Build();
                    DisplayMesh(targetMesh, transform);
                    icp = new MeshICP(sourceDMesh, targetTree);
                    icp.MaxIterations = 1;
                    savedTarget = true;
                }
            } else if (!isAligning)
            {
                Debug.Log("Starting alignment");
                AlignMeshes();
            }
        }
    }

    public Mesh CombineMeshes(List<MeshFilter> meshFilters)
    {
        CombineInstance[] combineInstances = new CombineInstance[meshFilters.Count];
        for (int i = 0; i < meshFilters.Count; i++)
        {
            combineInstances[i].mesh = meshFilters[i].sharedMesh;
            combineInstances[i].transform = meshFilters[i].transform.localToWorldMatrix;
        }

        // Combine meshes into a single mesh
        Mesh combinedMesh = new Mesh();
        combinedMesh.CombineMeshes(combineInstances, true, true);

        return combinedMesh;
    }

    public GameObject VolumeToMesh(OVRSceneVolume volume)
    {
        Mesh volumeMesh = new Mesh();
        float W = volume.Width;
        float H = volume.Height;
        float D = volume.Depth;
        Vector3 offset = volume.Offset - new Vector3(W/2, H/2, D);
        Transform volumeTransform = volume.transform;
        volumeMesh.vertices = new Vector3[] {new Vector3(0, 0, 0) + offset, new Vector3(W, 0, 0) + offset, new Vector3(W, H, 0) + offset, new Vector3(0, H, 0) + offset, 
                                            new Vector3(0, 0, D) + offset, new Vector3(W, 0, D) + offset, new Vector3(W, H, D) + offset, new Vector3(0, H, D) + offset};
        volumeMesh.triangles = new int[] {0, 1, 2, 2, 3, 0,
                                        1, 5, 6, 6, 2, 1,
                                        7, 6, 5, 5, 4, 7,
                                        4, 0, 3, 3, 7, 4,
                                        4, 5, 1, 1, 0, 4,
                                        3, 2, 6, 6, 7, 3};
        volumeMesh.triangles = volumeMesh.triangles.Reverse().ToArray();
        volumeMesh.RecalculateNormals();
        return DisplayMesh(volumeMesh, volumeTransform);
    }

    async void AlignMeshes()
    {
        isAligning = true;

        await Task.Run(() =>icp.Solve(true));

        sourceMeshObject.transform.position = new Vector3((float)icp.Translation.x, (float)icp.Translation.y, (float)icp.Translation.z);
        sourceMeshObject.transform.rotation = new Quaternion((float)icp.Rotation.x, (float)icp.Rotation.y, (float)icp.Rotation.z, (float)icp.Rotation.w);
        isAligning = false;
    }

    // Convert Unity.Mesh to g3.DMesh3
    private DMesh3 UnityMeshToDMesh(Mesh unityMesh)
    {
        Vector3[] vertices = unityMesh.vertices;
        int[] triangles = unityMesh.triangles;
        Vector3[] normals = unityMesh.normals;
        Vector3d[] g3Vertices = new Vector3d[vertices.Length];
        Vector3d[] g3Normals = new Vector3d[normals.Length];
        for (int i = 0; i < vertices.Length; i++)
        {
            g3Vertices[i] = new Vector3d(vertices[i].x, vertices[i].y, vertices[i].z);
        }
        for (int i = 0; i < normals.Length; i++)
        {
            g3Normals[i] = new Vector3d(normals[i].x, normals[i].y, normals[i].z);
        }
        DMesh3 dMesh = DMesh3Builder.Build(g3Vertices, triangles, g3Normals);
        return dMesh;
    }

    // Convert g3.DMesh3 to Unity.Mesh
    private Mesh DMeshToUnityMesh(DMesh3 dMesh)
    {
        Mesh unityMesh = new Mesh();
        Vector3d[] vertices = dMesh.Vertices().ToArray();
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
        material.color = new Color(0f, 0f, 1f, 0.75f);
        renderer.material = material;

        return meshObj;
    }
}
