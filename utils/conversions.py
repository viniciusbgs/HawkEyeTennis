
def convert_pixel_distance_to_meters(pixel_distance, reference_hight_in_meters, reference_hight_in_pixels):
    return (pixel_distance * reference_hight_in_meters) / reference_hight_in_pixels

def convert_meters_to_pixel_distance(meters, reference_hight_in_meters, reference_hight_in_pixels):
    return (meters * reference_hight_in_pixels) / reference_hight_in_meters