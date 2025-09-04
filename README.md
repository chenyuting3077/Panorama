# Panorama ¥þ´º¼v¹³«÷±µ

## Â²¤¶
¥þ´º¼v¹³¡]Panorama¡^±`¥Î©ó®·®»§¹¾ã³õ´º¡A¤×¨ä¬O¦b³æ±i·Ó¤ùµLªk²[»\¾ã­Ó´ºÆ[®É¡C¥»±M®×³z¹L **¼v¹³«÷±µ¡]Image Stitching¡^** §Þ³N¡A±N¦h±i¬Û¦P³õ´ºªº¼v¹³¦X¦¨¤@±i§¹¾ãªº¥þ´º¹Ï¡C  

¾ã­Ó¬yµ{¥i¤À¬°¤­­Ó¥D­n¶¥¬q¡G
1. **¯S¼xÂIÀË´ú¡]Detect Keypoints¡^**  
   - ¨Ï¥Î **Harris ¨¤ÂIÀË´ú¾¹** §ä¨ì¼v¹³¤¤­«­nªº¨¤ÂI¡C  
   - ±N RGB ¼v¹³Âà¬°¦Ç¶¥¡A¨Ã³z¹L Sobel Filter ­pºâ¼v¹³±è«×¡C
   
2. **¯S¼xÂI´y­z¡]Describe Keypoints¡^**  
   - ¨Ï¥Î **SIFT ´y­z¤l** ±N¨C­Ó¨¤ÂIªº§½³¡±è«×°T®§½s½X¦¨¦V¶q¡C  
   - ¥Í¦¨ 128 ºû descriptor¡A¨Ï¯S¼xÂI¨ã¦³±ÛÂà»P¤Ø«×¤£ÅÜ©Ê¡C
   
3. **¯S¼xÂI¤Ç°t¡]Match Keypoints¡^**  
   - ¤ñ¸û¨â±i¼v¹³ªº SIFT descriptor¡A³z¹L L2 ¶ZÂ÷§ä¨ì¤Ç°t¹ï¡C  
   - ³]©wìH­È¹LÂo¤£¥i¾aªº¤Ç°t¡C

4. **¤Ç°tµû¦ô¡]Evaluate Correspondences¡^**  
   - ¨Ï¥Î **RANSAC** ¿z¿ï¥X³ÌÃ­©wªº¤Ç°t¹ï¡C  
   - ®Ú¾Ú¤Ç°tÂI­pºâ **³æÀ³©Ê¯x°}¡]Homography¡^**¡A¥Î©ó¼v¹³¹ï»ô¡C

5. **¼v¹³ÅÜ´«»P¿Ä¦X¡]Transform & Blend¡^**  
   - §Q¥Î³æÀ³©Ê¯x°}±N¤@±i¼v¹³§ë¼v¨ì¥t¤@±i¼v¹³ªº¥­­±¤W¡C  
   - ¹ï­«Å|°Ï°ì¶i¦æ¿Ä¦X¡A¨Ï«÷±µ«áªº¥þ´º¹Ï¦ÛµM³s³e¡C# Panorama ?¨æ™¯å½±å??¼æŽ¥
