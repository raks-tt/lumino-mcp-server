(function () {
  'use strict';

  document.addEventListener('DOMContentLoaded', function () {
    var terminal = document.querySelector('.terminal');
    if (!terminal) return;

    var terminalBody = terminal.querySelector('.terminal-body');
    if (!terminalBody) return;

    var scenes = [
      {
        prompt: '$ ',
        command: 'lumino analyze-failed-pipeline --namespace ci-cd --run build-api-456',
        output: [
          '',
          '\u2500\u2500\u2500 Root Cause Analysis \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500',
          '',
          '  Pipeline:    build-api-456',
          '  Status:      FAILED',
          '  Duration:    3m 42s',
          '',
          '  Root Cause:  OOMKilled in build step (container exceeded 512Mi limit)',
          '  Secondary:   Image pull failed for registry.internal/base:latest (timeout)',
          '',
          '  Suggested Fix:',
          '    1. Increase memory limit to 1Gi for build-container',
          '    2. Verify registry connectivity and image tag',
          '    3. Add retry policy for transient pull failures',
          '',
          '  Confidence: 94%'
        ]
      },
      {
        prompt: '$ ',
        command: 'lumino resource-bottleneck-forecast --namespace production',
        output: [
          '',
          '\u2500\u2500\u2500 Resource Forecast (48-72h) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500',
          '',
          '  CPU Usage:     78% projected in 48h  (\u26a0 Warning)',
          '  Memory Usage:  CRITICAL in 72h       (\u2718 Action Required)',
          '  Disk I/O:      Stable                (\u2714 OK)',
          '',
          '  Trend:         +12% CPU / week,  +18% Memory / week',
          '',
          '  Recommendation:',
          '    1. Scale horizontally: add 2 replicas to api-gateway',
          '    2. Review memory leaks in payment-service (RSS growing 3%/day)',
          '    3. Consider HPA policy update: target CPU 60% \u2192 50%',
          '',
          '  Confidence: 87%'
        ]
      },
      {
        prompt: '$ ',
        command: 'lumino topology-map --namespace microservices',
        output: [
          '',
          '\u2500\u2500\u2500 Service Topology Map \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500',
          '',
          '  Services discovered:  12',
          '  Dependency chains:    3 critical paths',
          '',
          '  \u26a0 Circular dependency detected:',
          '    order-svc \u2192 inventory-svc \u2192 notification-svc \u2192 order-svc',
          '',
          '  Longest chain:',
          '    api-gw \u2192 auth-svc \u2192 user-svc \u2192 db-proxy \u2192 postgres (4 hops)',
          '',
          '  Single points of failure:',
          '    - auth-svc (9 dependents, no redundancy)',
          '    - db-proxy (6 dependents, single replica)',
          '',
          '  Confidence: 91%'
        ]
      }
    ];

    var CHAR_DELAY = 50;
    var POST_COMMAND_PAUSE = 2000;
    var OUTPUT_LINE_DELAY = 300;
    var SCENE_PAUSE = 4000;

    var currentScene = 0;
    var isVisible = true;
    var animationTimer = null;
    var isAnimating = false;

    // Create cursor element
    var cursor = document.createElement('span');
    cursor.className = 'terminal-cursor';
    cursor.textContent = '\u2588';

    // Intersection observer to pause when not visible
    var visibilityObserver = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        isVisible = entry.isIntersecting;
        if (isVisible && !isAnimating) {
          runScene(currentScene);
        }
      });
    }, { threshold: 0.1 });

    visibilityObserver.observe(terminal);

    function clearTerminal() {
      terminalBody.innerHTML = '';
    }

    function createLine() {
      var line = document.createElement('div');
      line.className = 'terminal-line';
      terminalBody.appendChild(line);
      return line;
    }

    function sleep(ms) {
      return new Promise(function (resolve) {
        animationTimer = setTimeout(resolve, ms);
      });
    }

    function typeCommand(line, text, charIndex) {
      return new Promise(function (resolve) {
        function typeNext() {
          if (!isVisible) {
            // Wait and retry when visible
            animationTimer = setTimeout(typeNext, 500);
            return;
          }

          if (charIndex < text.length) {
            line.textContent = text.substring(0, charIndex + 1);
            line.appendChild(cursor);
            charIndex++;
            animationTimer = setTimeout(typeNext, CHAR_DELAY);
          } else {
            resolve();
          }
        }
        typeNext();
      });
    }

    function revealOutputLines(lines) {
      return new Promise(function (resolve) {
        var i = 0;
        function revealNext() {
          if (!isVisible) {
            animationTimer = setTimeout(revealNext, 500);
            return;
          }

          if (i < lines.length) {
            var outputLine = createLine();
            outputLine.className = 'terminal-line terminal-output';
            outputLine.textContent = lines[i];
            i++;

            // Auto-scroll terminal body
            terminalBody.scrollTop = terminalBody.scrollHeight;

            animationTimer = setTimeout(revealNext, OUTPUT_LINE_DELAY);
          } else {
            resolve();
          }
        }
        revealNext();
      });
    }

    async function runScene(index) {
      if (isAnimating) return;
      isAnimating = true;

      var scene = scenes[index];

      clearTerminal();
      var commandLine = createLine();
      commandLine.className = 'terminal-line terminal-command';
      commandLine.textContent = scene.prompt;
      commandLine.appendChild(cursor);

      // Type the command
      await typeCommand(commandLine, scene.prompt + scene.command, scene.prompt.length);

      // Pause after command
      await sleep(POST_COMMAND_PAUSE);

      // Remove cursor from command line
      if (cursor.parentNode === commandLine) {
        commandLine.removeChild(cursor);
      }

      // Reveal output lines
      await revealOutputLines(scene.output);

      // Pause before next scene
      await sleep(SCENE_PAUSE);

      isAnimating = false;

      // Move to next scene
      currentScene = (currentScene + 1) % scenes.length;

      if (isVisible) {
        runScene(currentScene);
      }
    }

    // Start the animation
    if (isVisible) {
      runScene(currentScene);
    }
  });
})();
