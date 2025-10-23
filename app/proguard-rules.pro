# Add project specific ProGuard rules here.
# You can control the set of applied configuration files using the
# proguardFiles setting in build.gradle.
#
# For more details, see
#   http://developer.android.com/guide/developing/tools/proguard.html

# Uncomment this to preserve the line number information for
# debugging stack traces.
#-keepattributes SourceFile,LineNumberTable

# -renamesourcefileattribute SourceFile


########################################
# ONNX Runtime ProGuard Rules
########################################

# Keep all ONNX Runtime classes (uses JNI & reflection)
-keep class ai.onnxruntime.** { *; }
-dontwarn ai.onnxruntime.**

# Keep native library loaders (important for .so loading)
-keep class com.microsoft.onnxruntime.** { *; }
-dontwarn com.microsoft.onnxruntime.**

# Preserve model assets (ONNX models)
-keep class * extends java.io.InputStream { *; }
-keepattributes Signature

########################################
# Gson (Reflection-based JSON parsing)
########################################
-keep class com.google.gson.** { *; }
-keepattributes Signature
-keepattributes *Annotation*

########################################
# MediaPipe Tasks (optional but safe)
########################################
-keep class com.google.mediapipe.** { *; }
-dontwarn com.google.mediapipe.**

########################################
# TensorFlow Lite (optional for fallback)
########################################
-keep class org.tensorflow.** { *; }
-dontwarn org.tensorflow.**
