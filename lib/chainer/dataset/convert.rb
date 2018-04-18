module Chainer
  module Dataset
    module Convert
      def self.to_device(device, x)
        # TODO: support cuda
        x
      end

      def self.concat_examples(batch, device: nil, padding: nil)
        raise "batch is empty" if batch.size == 0
        first_elem = batch[0]

        if first_elem.kind_of?(Array)
          result = []
          unless padding.kind_of?(Array)
            padding = [padding] * first_elem.size
          end

          first_elem.size.times do |i|
            x = concat_arrays(batch.map { |b| b[i] }, padding[i])
            result.push(to_device(device, x))
          end

          return result
        else
          return to_device(device, concat_arrays(batch, padding))
        end
      end

      def self.concat_arrays(arrays, padding)
        unless arrays[0].kind_of?(Cumo::NArray)
          # [1, 2, 3, 4] => Cumo::Int32[1, 2, 3, 4]
          arrays = Cumo::NArray.cast(arrays)
          if padding
            return concat_arrays_with_padding(arrays, padding)
          end
          return arrays
        end

        if padding
          return concat_arrays_with_padding(arrays, padding)
        end

        # [Cumo::SFloat[1, 2], Cumo::SFloat[3, 4]]
        #  => Cumo::SFloat#shape=[2,2]
        # [[1, 2], [3, 4]]
        a = arrays.map{|arr| arr[:-, false]}
        a[0].concatenate(*a[1..-1])
      end

      def self.concat_arrays_with_padding(arrays, padding)
        if arrays[0].is_a? Cumo::NArray
          shape = Cumo::Int32.cast(arrays[0].shape)
          arrays[1..-1].each do |array|
            if Cumo::Bit.[](shape != array.shape).any?
              shape = Cumo::Int32.maximum(shape, array.shape)
            end
          end
        else # Integer
          shape = []
        end

        shape = shape.insert(0, arrays.size).to_a
        if arrays[0].is_a? Cumo::NArray
          result = arrays[0].class.new(shape).fill(padding)
        else # Integer
          result = Cumo::Int32.new(shape).fill(padding)
        end

        arrays.size.times do |i|
          src = arrays[i]
          if src.is_a? Cumo::NArray
            result[i, 0...src.shape[0], 0...src.shape[1]] = src
          else # Integer
            result[i] = src
          end
        end

        result
      end
    end
  end
end
