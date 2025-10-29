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
# Removed ONNX runtime keeps (PyTorch-only runtime)
########################################

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
# Removed TensorFlow Lite keeps (no longer used)
########################################
