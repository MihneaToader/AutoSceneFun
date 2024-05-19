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
    public OVRSceneManager sceneManager;
    public OVRPassthroughLayer passthrough;
    DMeshAABBTree3 targetTree;
    DMesh3 sourceDMesh;
    MeshICP icp;
    GameObject combinedMeshObj;
    const int vertexReductionFactor = 3;
    int frameWait = 0;
    int maxIcpIterations = 150;
    int icpIterations = 0;
    bool sceneModelLoaded = false;
    bool savedTarget = false;
    bool isAligning = false;
    public bool finishedAligning = false;

    void Start()
    {
        sourceMeshFilter = sourceMeshObject.GetComponent<MeshFilter>();
        sceneManager.gameObject.SetActive(true);
        passthrough.gameObject.SetActive(false);
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
                    // Compute target mesh tree
                    Mesh targetMesh = CombineMeshes(targetMeshes);
                    foreach (GameObject crtObj in toDestroy)
                        Destroy(crtObj);
                    DMesh3 targetDMesh = UnityMeshToDMesh(targetMesh);
                    targetTree = new DMeshAABBTree3(targetDMesh);
                    targetTree.Build();
                    combinedMeshObj = DisplayMesh(targetMesh, transform, "ComputedMesh");

                    // Compute simplified source mesh
                    sourceDMesh = UnityMeshToDMesh(sourceMeshFilter.sharedMesh);
                    Debug.Log("Initial vertex count: " + sourceDMesh.Vertices().ToList().Count);
                    Reducer r = new Reducer(sourceDMesh);
                    r.ReduceToVertexCount(sourceMeshFilter.sharedMesh.vertexCount / vertexReductionFactor);
                    Debug.Log("Reduced vertex count: " + sourceDMesh.Vertices().ToList().Count);
                    
                    // Initialize ICP with computed source mesh and target tree
                    icp = new MeshICP(sourceDMesh, targetTree);
                    icp.MaxIterations = 1;
                    icp.Translation = new Vector3d(sourceMeshObject.transform.position.x, sourceMeshObject.transform.position.y, sourceMeshObject.transform.position.z);
                    icp.Rotation = new Quaterniond(sourceMeshObject.transform.rotation.x, sourceMeshObject.transform.rotation.y, sourceMeshObject.transform.rotation.z, sourceMeshObject.transform.rotation.w);
                    icpIterations = 0;
                    savedTarget = true;
                }
            } else if (!isAligning && !finishedAligning)
            {
                Debug.Log("Starting alignment");
                AlignMeshes();
                icpIterations++;
                if (icpIterations >= maxIcpIterations)
                {
                    Debug.Log("Alignment finished");
                    finishedAligning = true;
                    sceneManager.gameObject.SetActive(false);
                    passthrough.gameObject.SetActive(true);
                    combinedMeshObj.GetComponent<Renderer>().enabled = !combinedMeshObj.GetComponent<Renderer>().enabled;
                    sourceMeshObject.GetComponent<Renderer>().enabled = !sourceMeshObject.GetComponent<Renderer>().enabled;
                    OVRSceneRoom sceneRoom = (OVRSceneRoom)Object.FindObjectOfType(typeof(OVRSceneRoom));
                    GameObject roomObject = sceneRoom.gameObject;
                    roomObject.SetActive(false);
                }
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
        return DisplayMesh(volumeMesh, volumeTransform, "Volume");
    }

    async void AlignMeshes()
    {
        isAligning = true;

        await Task.Run(() =>icp.Solve(true));
        
        // ICP may rotate about X or Z, which is not desired. Simply set X and Z rotation to 0 and update ICP internally for next iteration.
        Quaternion solvedRotationQuat = new Quaternion((float)icp.Rotation.x, (float)icp.Rotation.y, (float)icp.Rotation.z, (float)icp.Rotation.w);
        Vector3 eulerRep = solvedRotationQuat.eulerAngles;
        eulerRep.x = 0;
        eulerRep.z = 0;
        Quaternion fixedRotationQuat = Quaternion.Euler(eulerRep);
        icp.Rotation = new Quaterniond(fixedRotationQuat.x, fixedRotationQuat.y, fixedRotationQuat.z, fixedRotationQuat.w);
        sourceMeshObject.transform.position = new Vector3((float)icp.Translation.x, (float)icp.Translation.y, (float)icp.Translation.z);
        sourceMeshObject.transform.rotation = fixedRotationQuat;
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

    GameObject DisplayMesh(Mesh mesh, Transform parentTransform, string name)
    {
        GameObject meshObj = new GameObject(name);
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
