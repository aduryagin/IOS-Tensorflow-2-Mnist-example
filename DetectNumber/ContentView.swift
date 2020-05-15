import SwiftUI
import UIKit
import PencilKit
import CoreML
import Foundation
 
extension UIImage {
    func withBackground(color: UIColor, opaque: Bool = true) -> UIImage {
        UIGraphicsBeginImageContextWithOptions(size, opaque, scale)

        guard let ctx = UIGraphicsGetCurrentContext() else { return self }
        defer { UIGraphicsEndImageContext() }

        let rect = CGRect(origin: .zero, size: size)
        ctx.setFillColor(color.cgColor)
        ctx.fill(rect)
        ctx.concatenate(CGAffineTransform(a: 1, b: 0, c: 0, d: -1, tx: 0, ty: size.height))
        ctx.draw(cgImage!, in: rect)

        return UIGraphicsGetImageFromCurrentImageContext() ?? self
    }
    
    func resize(targetSize: CGSize) -> UIImage {
        let size = self.size
        
        let widthRatio  = targetSize.width  / size.width
        let heightRatio = targetSize.height / size.height
        
        var newSize: CGSize
        if widthRatio > heightRatio {
            newSize = CGSize(width: size.width * heightRatio, height: size.height * heightRatio)
        } else {
            newSize = CGSize(width: size.width * widthRatio,  height: size.height * widthRatio)
        }
        
        let rect = CGRect(x: 0, y: 0, width: newSize.width, height: newSize.height)
        
        UIGraphicsBeginImageContextWithOptions(newSize, false, 0)
        self.draw(in: rect)
        let newImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        
        return newImage!
    }
}

struct ContentView: View {
    @State var color = UIColor.white
    @State var demoImage: UIImage
    @State var predictNumber = -1
  
    var model = NumberDetectorModel()
    
    func preprocess(image: UIImage) -> MLMultiArray? {
        guard let result = try? MLMultiArray(shape: [1, 28, 28], dataType: .double) else {
            return nil
        }
        
        let data = image.cgImage!.dataProvider?.data
        let bytes = CFDataGetBytePtr(data)

        let bytesPerPixel = image.cgImage!.bitsPerPixel / image.cgImage!.bitsPerComponent
        
        for y in 0 ..< image.cgImage!.height {
            for x in 0 ..< image.cgImage!.width {
                let offset = (y * image.cgImage!.bytesPerRow) + (x * bytesPerPixel)
                let components = (r: bytes![offset], g: bytes![offset + 1], b: bytes![offset + 2])

                if x % 3 == 0 && y % 3 == 0 {
                    result[(y / 3) * 28 + (x / 3)] = NSNumber(value: (Float(components.r) + Float(components.g) + Float(components.b)) / 3)
                }
            }
        }

        return result
    }
    
    func getPrediction() {
        do {
            let image = PKCanvas.canvas?.drawing.image(from: CGRect(x: 0, y: 0, width: 300, height: 300), scale: 1)
            let newImage = image!.resize(targetSize: CGSize(width: 28, height: 28)).withBackground(color: .black)
            demoImage = newImage
            let color = preprocess(image: newImage)
            
            let input = NumberDetectorModelInput(flatten_input: color!)
            let prediction = try model.prediction(input: input)

            print(prediction.Identity)
            
            var maxValue: Float = 0.0;
            var maxIndex = 0;
            for index in 0..<prediction.Identity.count {
                if (Float(maxValue) < prediction.Identity[index].floatValue) {
                    maxIndex = index
                    maxValue = prediction.Identity[index].floatValue
                }
            }
            
            predictNumber = maxIndex
        } catch {
            print(error)
        }
        
    }
    
    var body: some View {
        VStack {
            PKCanvas(color: $color)
                .border(Color.black)
                .frame(width: 300, height: 300, alignment: .center)
            VStack {
//                Image(uiImage: demoImage).border(Color.red)
                Divider()
                Text(predictNumber == -1 ? "Draw something..." : "Maybe it's \(predictNumber)")
                Button("Clear Canvas"){
                    self.predictNumber = -1
                    PKCanvas.canvas!.drawing = PKDrawing()
                }
                Button("Prediction"){ self.getPrediction() }
            }
        }
    }
}

struct PKCanvas: UIViewRepresentable {
    class Coordinator: NSObject, PKCanvasViewDelegate {
        var pkCanvas: PKCanvas

        init(_ pkCanvas: PKCanvas) {
            self.pkCanvas = pkCanvas
        }
    }

    @Binding var color:UIColor

    static var canvas: PKCanvasView? = nil

    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }

    func makeUIView(context: Context) -> PKCanvasView {
        let canvas = PKCanvasView()
        canvas.backgroundColor = .black
        canvas.tool = PKInkingTool(.marker, color: color, width: 25)
        canvas.delegate = context.coordinator
        
        PKCanvas.canvas = canvas
        
        return canvas
    }

    func updateUIView(_ canvasView: PKCanvasView, context: Context) {
        
        canvasView.tool = PKInkingTool(.marker, color: color, width: 25)
    }
}
