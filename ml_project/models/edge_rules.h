#ifndef EDGE_RULES_H
#define EDGE_RULES_H

/* SubsiMonitor — on-device deterministic failsafe classifier
 * Distilled from the cloud RandomForest via a depth-4 decision tree
 * so it can run directly on the ESP32 with no ML runtime.
 * train acc=0.858  test acc=0.856
 */

// Auto-generated from a depth-limited DecisionTreeClassifier — see export_edge_rules.py
// Returns 1 if the edge node should raise a local CRITICAL flag, 0 otherwise.
int edge_risk_classify(float genergy, float gpuls, float gdenergy, float gdpuls) {
    if (genergy <= 18585.0000f) {
        if (gdpuls <= 65.5000f) {
            if (gdenergy <= -43.5000f) {
                if (gdenergy <= -45.5000f) {
                    return 0;  // leaf: [0.7683114222298102, 0.2316885777701901]
                } else {
                    return 1;  // leaf: [0.24903154399557278, 0.7509684560044273]
                }
            } else {
                if (gdpuls <= 3.5000f) {
                    return 0;  // leaf: [1.0, 0.0]
                } else {
                    return 0;  // leaf: [0.8522912811581588, 0.14770871884184195]
                }
            }
        } else {
            if (gdenergy <= 68.0000f) {
                return 0;  // leaf: [1.0, 0.0]
            } else {
                return 1;  // leaf: [0.30123583934088566, 0.6987641606591142]
            }
        }
    } else {
        if (gpuls <= 1254.5000f) {
            if (gpuls <= 156.0000f) {
                return 0;  // leaf: [1.0, 0.0]
            } else {
                if (genergy <= 21250.0000f) {
                    return 1;  // leaf: [0.2298524404086267, 0.7701475595913737]
                } else {
                    return 0;  // leaf: [0.5003278040719524, 0.499672195928038]
                }
            }
        } else {
            if (gdpuls <= 102.0000f) {
                if (gdpuls <= 20.0000f) {
                    return 1;  // leaf: [0.11503847658797453, 0.8849615234120259]
                } else {
                    return 1;  // leaf: [0.3577851396119266, 0.6422148603880744]
                }
            } else {
                if (gdenergy <= 159.5000f) {
                    return 0;  // leaf: [1.0, 0.0]
                } else {
                    return 0;  // leaf: [1.0, 0.0]
                }
            }
        }
    }
}

#endif // EDGE_RULES_H
