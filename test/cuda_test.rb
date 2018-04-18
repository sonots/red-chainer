# frozen_string_literal: true

require 'chainer'

class TestCuda < Test::Unit::TestCase
  def test_get_array_module_for_numpy()
    assert_equal(Cumo::NArray, Chainer::get_array_module(Cumo::NArray[]))
    assert_equal(Cumo::NArray, Chainer::get_array_module(Chainer::Variable.new(Cumo::NArray[])))
  end
end
