# frozen_string_literal: true

require 'chainer/functions/activation/tanh'

class Chainer::Functions::Activation::TanhTest < Test::Unit::TestCase
  data = {
    'test1' => {shape: [3, 2], dtype: xm::SFloat},
    'test2' => {shape: [], dtype: xm::SFloat},
    'test3' => {shape: [3, 2], dtype: xm::DFloat},
    'test4' => {shape: [], dtype: xm::DFloat}}

  def _setup(data)
    @shape = data[:shape]
    @dtype = data[:dtype]
    @dtype.srand(1) # To avoid false of "nearly_eq().all?", Use fixed seed value.
    @x = @dtype.new(@shape).rand(1) - 0.5
    @gy = @dtype.new(@shape).rand(0.2) - 0.1
    @check_backward_options = {}
  end

  def check_forward(x_data, use_cudnn: "always")
    x = Chainer::Variable.new(x_data)
    y = Chainer::Functions::Activation::Tanh.tanh(x)
    assert_equal(@dtype, y.data.class)
    y_expect = Chainer::Functions::Activation::Tanh.tanh(Chainer::Variable.new(@x))
    assert_true(y.data.nearly_eq(y_expect.data).all?)
  end

  data(data)
  def test_forward(data)
    _setup(data)
    check_forward(@x.dup)
  end

  def check_backward(x_data, gy_data, use_cudnn: "always")
    Chainer::check_backward(Chainer::Functions::Activation::Tanh.method(:tanh), x_data, gy_data, @check_backward_options)
  end

  data(data)
  def test_backward(data)
    _setup(data)
    check_backward(@x.dup, @gy.dup)
  end
end
