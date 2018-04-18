module Chainer
  # Gets an appropriate one from +Cumo::NArray+ or +Cumo::NArray+.
  #
  # This is almost equivalent to +Chainer::get_array_module+. The differences
  # are that this function can be used even if CUDA is not available and that
  # it will return their data arrays' array module for
  # +Chainer::Variable+ arguments.
  #
  # @param [Array<Chainer::Variable> or Array<Cumo::NArray> or Array<Cumo::NArray>] args Values to determine whether Cumo or Cumo should be used.
  # @return [Cumo::NArray] +Cumo::NArray+ or +Cumo::NArray+ is returned based on the types of
  #   the arguments.
  # @todo CUDA is not supported, yet.
  #
  def get_array_module(*args)
    return Cumo::NArray
  end
  module_function :get_array_module
end
