For today, 20251123, we are going to annotate second euljiro dataset.
For clairity, we need to define the annotation rule first.

1) Ascending
   * on the staircase: ascending stairs
   * on the platform: whole body(including more than a half of shoes) is appeared, and heading to the staircase.

2) Descending
   * on the staircase: passenger descending stairs
   * on the platform: staring train doors (1,2) and try to rush to the door.

3) Passing
   * the rest of them

4) etc.
   * passenger above of the smoke barrier is not annotated since it is not clear to classify.
   * if a passenger is in the bottom or left edge, we annotated it. However, if there is no person part in the edge, we don't annptated it. For example, if there is only a part of bag or clothes that is not contain human body part, we don't annotated it.

