FROM ubuntu:18.04

# packages
# ref. https://github.com/rbenv/ruby-build/wiki
RUN apt-get update \
  && apt-get install -y --no-install-recommends curl git \
  && apt-get install -y --no-install-recommends autoconf bison build-essential libssl-dev libyaml-dev libreadline6-dev zlib1g-dev libncurses5-dev libffi-dev libgdbm5 libgdbm-dev \
  && apt-get clean

# rbenv
ENV PATH /root/.rbenv/bin:$PATH
RUN git clone https://github.com/sstephenson/rbenv.git /root/.rbenv \
  && mkdir -p /root/.rbenv/plugins && git clone https://github.com/sstephenson/ruby-build.git /root/.rbenv/plugins/ruby-build \
  && echo -e 'export PATH=/root/.rbenv/bin:$PATH\neval "$(rbenv init -)"' >> /etc/profile.d/rbenv.sh

ENV CONFIGURE_OPTS --disable-install-doc
RUN bash -lc '~/.rbenv/plugins/ruby-build/bin/ruby-build 2.6.0-dev ~/.rbenv/versions/2.6.0-dev'

RUN git clone https://github.com/red-data-tools/red-chainer /root/red-chainer
