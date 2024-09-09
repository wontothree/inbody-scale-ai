# Person Segmentation

키 측정을 위한 Person Semantic Segmentation 코드입니다.

# Functions

|Function Name|Description|
|---|---|
|__init__|생성자 함수|
|resize_and_show|주어진 이미지를 원하는 크기로 조정하고, 조정된 이미지를 화면에 표시한다. 주어진 이미지를 지정된 크기로 조정하고 이를 화면에 표시하기 위한 함수이다.|
|**segment_image**|이미지 파일을 segmentation하고, 결과를 처리하여 출력 이미지를 만든다.|
|find_highest_pixel|주어진 카테고리 마스크에서 가장 높은 픽셀의 위치를 찾는다.|
|print_highest_pixel|주어진 이미지 파일을 segmentation하고 가장 높은 픽셀의 위치를 출력한다.|
|capture_and_segment|실시간으로 웹캠에서 프레임을 캡처하고, 's' 키가 눌릴 때마다 세그멘테이션을 수행한다.|
