#import <CoreML/CoreML.h>
#include <string.h>

// A-hn's Core ML program called from C, so two lanes (Neural Engine, GPU) run at once without Python's GIL.

void *ahn_open(const char *package, const char *compiled_path, int units) {
    @autoreleasepool {
        NSFileManager *files = [NSFileManager defaultManager];
        NSURL *compiled = [NSURL fileURLWithPath:[NSString stringWithUTF8String:compiled_path]];
        NSError *error = nil;
        if (![files fileExistsAtPath:compiled.path]) {
            NSURL *made = [MLModel compileModelAtURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:package]] error:&error];
            if (!made) { NSLog(@"ahn: %@", error); return NULL; }
            if (![files moveItemAtURL:made toURL:compiled error:&error]) [files removeItemAtURL:made error:nil];
        }
        MLModelConfiguration *configuration = [MLModelConfiguration new];
        configuration.computeUnits = (MLComputeUnits)units;
        MLModel *model = [MLModel modelWithContentsOfURL:compiled configuration:configuration error:&error];
        if (!model) { NSLog(@"ahn: %@", error); return NULL; }
        return (__bridge_retained void *)model;
    }
}

// One tile: tiles [1, depth, side, side] float32 -> logits [side * side] float32. 0 on success.
int ahn_predict(void *engine, float *tiles, int depth, int side, float *logits) {
    @autoreleasepool {
        NSError *error = nil;
        MLMultiArray *input = [[MLMultiArray alloc] initWithDataPointer:tiles
                                                                  shape:@[ @1, @(depth), @(side), @(side) ]
                                                               dataType:MLMultiArrayDataTypeFloat32
                                                                strides:@[ @(depth * side * side), @(side * side), @(side), @1 ]
                                                            deallocator:nil
                                                                  error:&error];
        if (!input) return 1;
        MLDictionaryFeatureProvider *given = [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{@"tiles" : input} error:&error];
        if (!given) return 2;
        id<MLFeatureProvider> found = [(__bridge MLModel *)engine predictionFromFeatures:given error:&error];
        if (!found) { NSLog(@"ahn: %@", error); return 3; }
        MLMultiArray *out = [found featureValueForName:@"logits"].multiArrayValue;
        if (out.count != side * side) return 4;
        __block int failed = 0;
        [out getBytesWithHandler:^(const void *bytes, NSInteger size) {
          NSInteger row = out.strides[out.strides.count - 2].integerValue;
          if (out.dataType == MLMultiArrayDataTypeFloat32) {
              for (int y = 0; y < side; y++) memcpy(logits + (size_t)y * side, (const float *)bytes + (size_t)y * row, (size_t)side * sizeof(float));
          } else if (out.dataType == MLMultiArrayDataTypeFloat16) {
              for (int y = 0; y < side; y++)
                  for (int x = 0; x < side; x++) logits[(size_t)y * side + x] = (float)((const _Float16 *)bytes)[(size_t)y * row + x];
          } else {
              failed = 5;
          }
        }];
        return failed;
    }
}
