using System;
using System.IO;
using UnityEngine;

public enum HandInfoFrequency
{
    None,
    Once,
    Repeat
}

public class HandDebugSkeletonInfo : MonoBehaviour
{
    [SerializeField]
    private OVRHand hand;

    [SerializeField]
    private OVRSkeleton handSkeleton;

    [SerializeField]
    private HandInfoFrequency handInfoFrequency = HandInfoFrequency.Once;

    private bool handInfoDisplayed = false;

    private bool pauseDisplay = false;

    private void Awake()
    {
        if(!hand) hand = GetComponent<OVRHand>();
        if(!handSkeleton) handSkeleton = GetComponent<OVRSkeleton>();
    }

    private void Update(){
#if UNITY_EDITOR
    if(Input.GetKeyDown(KeyCode.Space)) pauseDisplay = !pauseDisplay;
#endif
    if(hand.IsTracked && !pauseDisplay)
    {
        if (handInfoFrequency == HandInfoFrequency.Once && !handInfoDisplayed)
        {
            DisplayBoneInfo();
            handInfoDisplayed = true;
        }
        else if (handInfoFrequency == HandInfoFrequency.Repeat)
        {
            DisplayBoneInfo();
        }
    }
}
private void DisplayBoneInfo()
{
    string tmp_text = "";
    string _path;
   
    foreach (var bone in handSkeleton.Bones)
    {
        tmp_text += $"{handSkeleton.GetSkeletonType()}: boneId -> {bone.Id} pos -> {bone.Transform.position} rot -> {bone.Transform.rotation}" + "\n";
        // Debug.Log($"{handSkeleton.GetSkeletonType()}: boneId -> {bone.Id} pos -> {bone.Transform.position}");
    }
    tmp_text += $"{handSkeleton.GetSkeletonType()}: num of bones -> {handSkeleton.GetCurrentNumBones()}" + "\n";
    tmp_text += $"{handSkeleton.GetSkeletonType()}: num of skinnable bones -> {handSkeleton.GetCurrentNumSkinnableBones()}" + "\n";
    tmp_text += $"{handSkeleton.GetSkeletonType()}: start bone id -> {handSkeleton.GetCurrentStartBoneId()}" + "\n";
    tmp_text += $"{handSkeleton.GetSkeletonType()}: end bone id -> {handSkeleton.GetCurrentEndBoneId()} "+ "\n";
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
