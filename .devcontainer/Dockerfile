FROM python:3.10.1-bullseye

RUN echo 'root:root' | chpasswd
ENV SSH_AUTH_SOCK=/ssh-agent

# --- Create non-root user with the ability to use sudo --- #
ARG USERNAME=developer
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# For RlBerry display
RUN sudo apt-get install -y xvfb
RUN sudo apt-get install -y python3-opengl
RUN sudo apt-get install -y ffmpeg

RUN mkdir /home/developer/ReinforcementLearningWithDemonstration/ 
WORKDIR /home/developer/ReinforcementLearningWithDemonstration/

USER $USERNAME