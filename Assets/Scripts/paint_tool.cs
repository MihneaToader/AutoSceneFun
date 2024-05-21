using UnityEngine;

public class TextureColorChanger : MonoBehaviour
{
    public Camera camera;
    public Material materialToChange;

    private bool isQPressed = false;

    public int brushSize = 5;

    public  Color color = Color.red;

    Texture2D originalTexture = null;

    void Update()
    {
        // Check if the 'Q' key is pressed
        if (Input.GetKeyDown(KeyCode.Q))
        {
            isQPressed = true;
        }
        else if (Input.GetKeyUp(KeyCode.Q))
        {
            isQPressed = false;
        }

        // Change the texture color while the 'Q' key is pressed
        if (isQPressed)
        {
            // Cast a ray from the camera to the mouse position
            RaycastHit hit;
            Ray ray = camera.ScreenPointToRay(Input.mousePosition);
            if (Physics.Raycast(ray, out hit))
            {
                // Get the mesh renderer of the hit object
                MeshRenderer meshRenderer = hit.collider.GetComponent<MeshRenderer>();
                if (meshRenderer != null)
                {
                    // Get the original texture
                    if (originalTexture == null)
                    {
                        Texture2D originalTexture = (Texture2D)meshRenderer.material.mainTexture;
                    }

                    // Create a copy of the original texture
                    Texture2D copiedTexture = new Texture2D(originalTexture.width, originalTexture.height, TextureFormat.RGBA32, false);
                    copiedTexture.SetPixels(originalTexture.GetPixels());
                    copiedTexture.Apply();

                    // Set the color of the copied texture to red at the hit point
                    int xCoord = Mathf.FloorToInt(hit.textureCoord.x * originalTexture.width);
                    int yCoord = Mathf.FloorToInt(hit.textureCoord.y * originalTexture.height);
                    for (int i = -brushSize; i < brushSize; i++)
                    {
                        for (int j = -brushSize; j < brushSize; j++)
                        {
                            copiedTexture.SetPixel(xCoord + i, yCoord + j, color);
                        }
                    }
             
                    copiedTexture.Apply();

                    // Assign the copied texture to the same material
                    meshRenderer.material.mainTexture = copiedTexture;
                    originalTexture = copiedTexture;
                }
            }
        }
        else{
            // Cast a ray from the camera to the mouse position
            RaycastHit hit;
            Ray ray = camera.ScreenPointToRay(Input.mousePosition);
            if (Physics.Raycast(ray, out hit))
            {
                // Get the mesh renderer of the hit object
                MeshRenderer meshRenderer = hit.collider.GetComponent<MeshRenderer>();
                if (meshRenderer != null)
                {
                    // Get the original texture
                    if (originalTexture == null)
                    {
                        originalTexture = (Texture2D)meshRenderer.material.mainTexture;
                    }
                    Texture2D previewTexture = originalTexture;

                    // Create a copy of the original texture
                    Texture2D preview = new Texture2D(previewTexture.width, previewTexture.height, TextureFormat.RGBA32, false);
                    preview.SetPixels(previewTexture.GetPixels());
                    preview.Apply();

                    // Set the color of the copied texture to red at the hit point
                    int xCoord = Mathf.FloorToInt(hit.textureCoord.x * previewTexture.width);
                    int yCoord = Mathf.FloorToInt(hit.textureCoord.y * previewTexture.height);
                    for (int i = -brushSize; i < brushSize; i++)
                    {
                        for (int j = -brushSize; j < brushSize; j++)
                        {
                            preview.SetPixel(xCoord + i, yCoord + j, Color.white);
                        }
                    }

                    preview.Apply();

                    // Assign the copied texture to the same material
                    meshRenderer.material.mainTexture = preview;
                }
            }

        }
        
    }
}
